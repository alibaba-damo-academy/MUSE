"""
This file is used to eval models on linear/knn.

Eval Dataset:
1. BRCAM2C
2. PUMA
3. OCELOT

for each dataset:
1. define the loaded subdataset based on mpp (0.25, 0.50), patch size (224, 512, 1024)
2. load the target subdataset

for each model:
1. extract dense cell features based on the loaded subdataset (image & cell coordinates)
2. finetune the linear classification model (linear) or knn model (knn)
"""

import os
import tqdm
import random
import shutil
import argparse
import torch
from torch import nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import ModelFacotry

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--subdataset_name', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)

parser.add_argument('--eval_type', type=str, default='linear')

# ==== only for linear
parser.add_argument('--eval_epochs', type=int, default=100)
parser.add_argument('--eval_bs', type=int, default=256)
parser.add_argument('--eval_lr', type=float, default=0.01)

# ==== only for knn
parser.add_argument('--k', type=int, default=[10, 20, 100, 200, 500], nargs='+', help='k values for knn')
parser.add_argument('--temperature', type=float, default=0.07)
args = parser.parse_args()


class FeatureExtractDataset(Dataset):

    def __init__(self, loaded_data):
        super(FeatureExtractDataset, self).__init__()

        self.patches = loaded_data['patches']
        self.anns = loaded_data['anns']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sample_names = list(self.patches.keys())

    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, idx):
        # get sample
        sample_name = self.sample_names[idx]
        patch = self.patches[sample_name]
        patch = self.transform(patch)

        ann = self.anns[sample_name]

        # input image, cell coordinates, cell labels, sample name
        return patch, torch.tensor(ann[:, :2], dtype=torch.float), torch.tensor(ann[:, 2], dtype=torch.long), sample_name
    
    @staticmethod
    def collate_fn(batch):
        images, coords, labels, sample_names = [[] for _ in range(4)]
        for x in batch:
            images.append(x[0])
            coords.append(x[1])
            labels.append(x[2])
            sample_names.append(x[3])
        return torch.stack(images), coords, labels, sample_names


class FeatureEvalDataset(Dataset):

    def __init__(self, all_feats, all_labels):
        super().__init__()

        self.all_feats = all_feats
        self.all_labels = all_labels
    
    def __len__(self):
        return len(self.all_feats)
    
    def __getitem__(self, idx):
        return self.all_feats[idx], self.all_labels[idx]


def extract_feature():

    # init data, dataset and dataloader
    eval_data = np.load(f'./dataset/{args.dataset_name}/{args.subdataset_name}.npy', allow_pickle=True).item()  # load data

    feature_extract_ds = FeatureExtractDataset(eval_data)
    feature_extract_dl = DataLoader(
        feature_extract_ds,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=feature_extract_ds.collate_fn
    )

    # init model
    model = ModelFacotry.get_model(args.model_name)
    model.cuda()
    model.eval()

    # inference
    dense_feats = dict()
    dense_labels = dict()
    with torch.no_grad():
        for bs_id, bs in tqdm.tqdm(enumerate(feature_extract_dl)):
            input_images, gt_coords, gt_labels, sample_names = bs
            input_images = input_images.cuda()

            pred_feats = model(input_images, gt_coords)
            for sample_id, pred_feat in enumerate(pred_feats):
                fully_sample_name = sample_names[sample_id]
                split = fully_sample_name.split('_')[0]
                sample_name = '_'.join(fully_sample_name.split('_')[1:])
                if split not in dense_feats:
                    dense_feats[split] = {}
                    dense_labels[split] = {}

                dense_feats[split][sample_name] = pred_feat.detach().cpu()
                dense_labels[split][sample_name] = gt_labels[sample_id].detach()

                assert dense_feats[split][sample_name].shape[0] == dense_labels[split][sample_name].shape[0], \
                    'the number of dense feats is not equal to the number of gt labels, please check'
    
    # fully check
    # the number of dense feats 
    print('extract dense features done...')

    return dense_feats, dense_labels, model.out_dim


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def build_eval_ds_dl(eval_feats, eval_labels, shuffle=False, drop_last=False):
    _ds = FeatureEvalDataset(eval_feats, eval_labels)
    _dl = DataLoader(
        _ds,
        batch_size=args.eval_bs,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last,
        pin_memory=True
    )
    return _dl


def run_eval_linear(eval_feats, eval_labels, dense_feat_dim, fold_list):

    output_folder = f'./outputs/{args.model_name}_{args.dataset_name}_{args.subdataset_name}_{args.eval_type}_{args.seed}'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    log_file = f'{output_folder}/eval_linear.txt'

    # == build the linear model ==
    if args.dataset_name == 'BRCAM2C':
        n_cell_type = 3
    elif args.dataset_name == 'OCELOT':
        n_cell_type = 2
    elif args.dataset_name == 'PUMA':
        # coarse / fine
        if 'fine' in args.subdataset_name:
            n_cell_type = 10
        else:
            n_cell_type = 3
    else:
        raise NotImplementedError(f'{args.dataset_name} is not supported.')

    classifier = LinearClassifier(dense_feat_dim, n_cell_type)
    classifier.cuda()

    # == build the train/val/test dataset & loader ==
    train_dl = build_eval_ds_dl(
        eval_feats[fold_list[0]],
        eval_labels[fold_list[0]],
        shuffle=True,
        drop_last=True
    )

    val_dl = build_eval_ds_dl(
        eval_feats[fold_list[1]],
        eval_labels[fold_list[1]],
        shuffle=False,
        drop_last=False
    )

    test_dl = build_eval_ds_dl(
        eval_feats[fold_list[2]],
        eval_labels[fold_list[2]],
        shuffle=False,
        drop_last=False
    )

    # == build optim / sch ==
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.eval_lr,
        momentum=0.9,
        weight_decay=0
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.eval_epochs,
        eta_min=0
    )

    criterion = nn.CrossEntropyLoss()

    # == train/val loop ==
    best_acc = 0
    print('Start training...')
    for epoch in range(args.eval_epochs):

        classifier.train()

        epoch_gt = list()
        epoch_pred = list()
        for bs_id, bs in enumerate(train_dl):
            in_feats, labels = bs
            in_feats = in_feats.cuda()
            labels = labels.cuda()

            pred = classifier(in_feats)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_gt.append(labels.detach().cpu())
            epoch_pred.append(pred.detach().cpu())

            if bs_id % 20 == 0:
                print('Epoch: {:03d} | Batch: {:03d} | Loss: {:.4f} | LR: {:.4e}'.format(epoch, bs_id, loss.item(), optimizer.param_groups[0]['lr']))

        sch.step()

        epoch_gt = torch.cat(epoch_gt)
        epoch_pred = torch.cat(epoch_pred)
        epoch_loss = criterion(epoch_pred, epoch_gt).item()

        with torch.no_grad():
            val_gt = list()
            val_pred = list()
            classifier.eval()
            for val_bs_id, val_bs in enumerate(val_dl):

                val_feats, val_labels = val_bs
                val_feats = val_feats.cuda()

                pred = classifier(val_feats)
                pred = torch.argmax(pred, dim=1)

                val_gt.append(val_labels.cpu())
                val_pred.append(pred.cpu())
            
            val_gt = torch.cat(val_gt)
            val_pred = torch.cat(val_pred)
            val_acc = torch.sum(val_gt == val_pred).item() / len(val_gt)

        # == test ==
        # load the best
        with torch.no_grad():
            classifier.eval()

            test_gt = list()
            test_pred = list()
            for i, (feats, labels) in enumerate(test_dl):

                pred = classifier(feats.cuda())
                pred = torch.argmax(pred, dim=1)

                test_gt.append(labels.cpu())
                test_pred.append(pred.cpu())
            
            test_gt = torch.cat(test_gt)
            test_pred = torch.cat(test_pred)
            test_acc = torch.sum(test_gt == test_pred).item() / len(test_gt)
        
        epoch_str = f'Epoch {epoch} | EpochTrainLoss : {epoch_loss:.4f} | ValAcc : {val_acc:.4f}\n'
        print(epoch_str, f'TestAcc: {test_acc:.4f}')
        with open(log_file, 'a') as f:
            f.write(epoch_str)

        if best_acc <= val_acc:
            best_acc = val_acc
            # save the model
            torch.save(
                classifier.state_dict(),
                f"{output_folder}/best_linear.pth"
            )
    
    # == test ==
    # load the best
    classifier.load_state_dict(torch.load(f"{output_folder}/best_linear.pth"))
    with torch.no_grad():
        classifier.eval()

        test_gt = list()
        test_pred = list()
        for i, (feats, labels) in enumerate(test_dl):

            pred = classifier(feats.cuda())
            pred = torch.argmax(pred, dim=1)

            test_gt.append(labels.cpu())
            test_pred.append(pred.cpu())
        
        test_gt = torch.cat(test_gt)
        test_pred = torch.cat(test_pred)
        test_acc = torch.sum(test_gt == test_pred).item() / len(test_gt)

    # save the output and gt
    np.save(f"{output_folder}/linear_eval_gt.npy", test_gt.numpy())
    np.save(f"{output_folder}/linear_eval_pred.npy", test_pred.numpy())

    print(f"TestAcc: {test_acc:.4f}")
    with open(log_file, 'a') as f:
        f.write(f"TestAcc: {test_acc:.4f}")

    return test_acc


def run_eval_knn(eval_feats, eval_labels, dense_feat_dim, fold_list):

    output_folder = f'./outputs/{args.model_name}_{args.dataset_name}_{args.subdataset_name}_{args.eval_type}_{args.seed}'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    log_file = f'{output_folder}/eval_knn.txt'

    if args.dataset_name == 'BRCAM2C':
        n_cell_type = 3
    elif args.dataset_name == 'OCELOT':
        n_cell_type = 2
    elif args.dataset_name == 'PUMA':
        # coarse / fine
        if 'fine' in args.subdataset_name:
            n_cell_type = 10
        else:
            n_cell_type = 3
    else:
        raise NotImplementedError(f'{args.dataset_name} is not supported.')

    knn_acc = list()
    for k in args.k:
        
        # ========================== ATTENTION ==========================
        # for KNN, should normalize
        top1 = knn_classifier(
            train_features=nn.functional.normalize(eval_feats[fold_list[0]], dim=1, p=2),
            train_labels=eval_labels[fold_list[0]],
            test_features=nn.functional.normalize(eval_feats[fold_list[2]], dim=1, p=2),
            test_labels=eval_labels[fold_list[2]],
            k=k,
            T=args.temperature,
            num_classes=n_cell_type,
        )

        knn_acc_str = f'KNN k={k} Top1: {top1}'
        knn_acc.append(top1)
        print(knn_acc_str)
        with open(log_file, 'a') as f:
            f.write(f'{knn_acc_str}\n')
    
    knn_best_acc_str = f'KNN Best Top1: {max(knn_acc)}'
    print(knn_best_acc_str)
    with open(log_file, 'a') as f:
        f.write(f'{knn_best_acc_str}\n')


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        # top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    # top5 = top5 * 100.0 / total
    return top1


if __name__ == '__main__':

    # multiversion timm switch
    if args.model_name in [
        'scr_r50',
        'dino_r50',
        'dino_vit-s-16', 'dino_vit-b-16', 
        'sup_r50', 'sup_vit-s-16', 'sup_vit-b-16',
        'mae_vit-b',
        'ibot_vit-s-16', 'ibot_vit-b-16',
        'dinov2_vit-s', 'dinov2_vit-b',
        'dinov2_vit-b-path', 'dinov2_vit-s-path',
        'kang_bench_r50', 'kang_bench_vit-s-16',
        'ctrans', 'chief',
        'muse_r50', 'lfov_muse_r50',
        'muse_vit-s-16', 'lfov_muse_vit-s-16',
        'muse_vit-b-16', 'lfov_muse_vit-b-16'
    ]:
        # use 0.6.13
        import sys
        sys.path.append('./multiversion_timm/timm_0613/')
        import timm
        print(f"Using timm version: {timm.__version__}")
    elif args.model_name in ['uni', 'conch', 'gigapath']:
        # use 1.0.15
        import sys
        sys.path.append('./multiversion_timm/timm_1015/')
        import timm
        print(f"Using timm version: {timm.__version__}")
    else:
        raise NotImplementedError("model_name not supported")

    # == seed all ==
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # == extractor dense features ==
    dense_feats, dense_labels, out_dim = extract_feature()

    # == build the eval data ==
    eval_feats = {k: torch.cat(list(v.values()), dim=0) for k, v in dense_feats.items()}
    eval_labels = {k: torch.cat(list(v.values()), dim=0) for k, v in dense_labels.items()}
    
    for fold_name, fold_feat in eval_feats.items():
        print(f"eval_feats[{fold_name}]: {fold_feat.shape}")
        print(f"eval_labels[{fold_name}]: {eval_labels[fold_name].shape}")

    # train / val / test
    if args.eval_type == 'linear':
        print('=====================')
        print('Linear Evaluation')
        print('=====================')

        acc = run_eval_linear(eval_feats, eval_labels, out_dim, ['train', 'val', 'test'])
    elif args.eval_type == 'knn':
        print('=====================')
        print('KNN Evaluation')
        print('=====================')
        run_eval_knn(eval_feats, eval_labels, out_dim, ['train', 'val', 'test'])
    else:
        raise ValueError('Invalid eval type')
