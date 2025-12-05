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

# ==== only for linear
parser.add_argument('--eval_epochs', type=int, default=10)
parser.add_argument('--eval_bs', type=int, default=32)
parser.add_argument('--eval_lr', type=float, default=1e-5)

args = parser.parse_args()


class FinetuningModel(nn.Module):

    def __init__(self, model_name, num_labels=1000):

        super(FinetuningModel, self).__init__()

        self.backbone = ModelFacotry.get_model(model_name)
        self.linear = LinearClassifier(self.backbone.out_dim, num_labels)

        # we init the linear with the model of linear eval
        linear_folder = f'./outputs/{args.model_name}_{args.dataset_name}_{args.subdataset_name}_linear_{args.seed}'
        assert os.path.exists(linear_folder), f'{linear_folder} does not exist'
        self.linear.load_state_dict(torch.load(os.path.join(linear_folder, 'best_linear.pth')))
    
    def forward(self, input_images, cell_coords):
        pred_feats = self.backbone(input_images, cell_coords)
        pred_logits = self.linear(torch.cat(pred_feats, dim=0))
        return pred_logits


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

        # norm = torch.norm(x, p=2, dim=-1)
        # print(torch.std(norm), torch.mean(norm), torch.max(norm), torch.min(norm))
        # x = torch.nn.functional.normalize(x, dim=1, p=2)

        # linear layer
        return self.linear(x)


class FinetuningDataset(Dataset):

    def __init__(self, loaded_data, split):
        super(FinetuningDataset, self).__init__()

        self.patches = loaded_data['patches']
        self.anns = loaded_data['anns']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        sample_names = list(self.patches.keys())

        # fileter sample_names with split
        self.sample_names = [sample_name for sample_name in sample_names if sample_name.split('_')[0] == split]

    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, idx):
        # get sample
        sample_name = self.sample_names[idx]
        patch = self.patches[sample_name]
        patch = self.transform(patch)

        ann = self.anns[sample_name]

        # input image, cell coordinates, cell labels, sample name
        return patch, torch.tensor(ann[:, :2], dtype=torch.float), torch.tensor(ann[:, 2], dtype=torch.long)
    
    @staticmethod
    def collate_fn(batch):
        images, coords, labels = [[] for _ in range(3)]
        for x in batch:
            images.append(x[0])
            coords.append(x[1])
            labels.append(x[2])
        return torch.stack(images), coords, labels


def build_eval_ds_dl(split, eval_data, shuffle=False, drop_last=False):
    _ds = FinetuningDataset(eval_data, split)
    _dl = DataLoader(
        _ds,
        batch_size=args.eval_bs,
        shuffle=shuffle,
        num_workers=4,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=FinetuningDataset.collate_fn
    )
    return _dl


def run_eval_finetuning(fold_list):

    # init data, dataset and dataloader
    eval_data = np.load(f'./dataset/{args.dataset_name}/{args.subdataset_name}.npy', allow_pickle=True).item()  # load data

    # == build the train/val/test dataset & loader ==
    train_dl = build_eval_ds_dl(
        fold_list[0],
        eval_data,
        shuffle=True,
        drop_last=True
    )
    val_dl = build_eval_ds_dl(
        fold_list[1],
        eval_data,
        shuffle=False,
        drop_last=False
    )
    test_dl = build_eval_ds_dl(
        fold_list[2],
        eval_data,
        shuffle=False,
        drop_last=False
    )

    # == build output folder ==
    output_folder = f'./outputs/{args.model_name}_{args.dataset_name}_{args.subdataset_name}_finetuning_{args.seed}'
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

    model = FinetuningModel(args.model_name, n_cell_type)
    model.cuda()

    # == build optim / sch ==
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.eval_lr,
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

        model.train()

        epoch_gt = list()
        epoch_pred = list()
        for bs_id, bs in enumerate(train_dl):
            input_images, gt_coords, gt_labels = bs
            input_images = input_images.cuda()
            gt_labels = torch.cat(gt_labels, dim=0).cuda()
            pred_logits = model(input_images, gt_coords)

            loss = criterion(pred_logits, gt_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_gt.append(gt_labels.detach().cpu())
            epoch_pred.append(pred_logits.detach().cpu())

            if bs_id % 20 == 0:
                print('Epoch: {:03d} | Batch: {:03d} | Loss: {:.4f} | LR: {:.4e}'.format(epoch, bs_id, loss.item(), optimizer.param_groups[0]['lr']))

        sch.step()

        epoch_gt = torch.cat(epoch_gt)
        epoch_pred = torch.cat(epoch_pred)
        epoch_loss = criterion(epoch_pred, epoch_gt).item()

        with torch.no_grad():
            val_gt = list()
            val_pred = list()
            model.eval()
            for val_bs_id, val_bs in enumerate(val_dl):
                
                val_images, val_coords, val_labels = val_bs
                val_images = val_images.cuda()
                val_labels = torch.cat(val_labels, dim=0).cuda()
                pred_logits = model(val_images, val_coords)
                pred = torch.argmax(pred_logits, dim=1)

                val_gt.append(val_labels.cpu())
                val_pred.append(pred.cpu())
            
            val_gt = torch.cat(val_gt)
            val_pred = torch.cat(val_pred)
            val_acc = torch.sum(val_gt == val_pred).item() / len(val_gt)

        # == test ==
        # load the best
        with torch.no_grad():
            model.eval()

            test_gt = list()
            test_pred = list()
            for i, (test_images, test_coords, test_labels) in enumerate(test_dl):

                test_images = test_images.cuda()
                test_labels = torch.cat(test_labels, dim=0).cuda()

                pred_logits = model(test_images, test_coords)
                pred = torch.argmax(pred_logits, dim=1)

                test_gt.append(test_labels.cpu())
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
                model.state_dict(),
                f"{output_folder}/best_model.pth"
            )
    
    # == test ==
    # load the best
    model.load_state_dict(torch.load(f"{output_folder}/best_model.pth"))
    with torch.no_grad():
        model.eval()

        test_gt = list()
        test_pred = list()
        for i, (test_images, test_coords, test_labels) in enumerate(test_dl):

            test_images = test_images.cuda()
            test_labels = torch.cat(test_labels, dim=0).cuda()

            pred_logits = model(test_images, test_coords)
            pred = torch.argmax(pred_logits, dim=1)

            test_gt.append(test_labels.cpu())
            test_pred.append(pred.cpu())
        
        test_gt = torch.cat(test_gt)
        test_pred = torch.cat(test_pred)
        test_acc = torch.sum(test_gt == test_pred).item() / len(test_gt)

    # save the output and gt
    np.save(f"{output_folder}/finetuning_eval_gt.npy", test_gt.numpy())
    np.save(f"{output_folder}/finetuning_eval_gt.npy", test_pred.numpy())

    print(f"TestAcc: {test_acc:.4f}")
    with open(log_file, 'a') as f:
        f.write(f"TestAcc: {test_acc:.4f}")

    return test_acc


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
    elif args.model_name in ['uni', 'uni2', 'conch', 'gigapath']:
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

    # train / val / test
    print('=====================')
    print('Finetuning Evaluation')
    print('=====================')

    acc = run_eval_finetuning(['train', 'val', 'test'])
