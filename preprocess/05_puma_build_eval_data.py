"""
This file is used to build the data for linear/knn evaluation

args:
- puma_folder: the folder of PUMA
- out_folder: the folder to save the data

base on https://puma.grand-challenge.org/dataset/, the default scale is 40x

This file will build six subdataset for linear/knn evaluation:
- 20x, 224: 20x resolution, 224x224 image, 224x224 annotation
- 20x, 512: 20x resolution, 512x512 image, 224x224 annotation
- 20x, 1024: 20x resolution, 1024x1024 image, 224x224 annotation
- 40x, 224: 40x resolution, 224x224 image, 224x224 annotation
- 40x, 512: 40x resolution, 512x512 image, 224x224 annotation
- 40x, 1024: 40x resolution, 1024x1024 image, 224x224 annotation

For each raw annotated sample, we crop it to small patches, and save them to the out_folder.
"""
import os
import cv2
import glob
import json
import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--puma_folder', type=str, default='', help='Path to the puma folder.')
parser.add_argument('--out_folder', type=str, default='', help='Path to the output folder.')
args = parser.parse_args()


def ana_split_file(input_dict, split_file, split_name):

    with open(split_file, 'r') as f:
        split_dict = json.load(f)
        file_names = [file_name[7:-5] for file_name in list(split_dict['anno'].keys())]

        for file_name in file_names:
            assert file_name not in input_dict, 'file name already exists'
            input_dict[file_name] = split_name
    
    return input_dict


def compute_center(point_list):
    contour = np.array(point_list, dtype=np.int32).reshape((-1, 1, 2))
    M = cv2.moments(contour)

    if M['m00'] != 0:
        # 质心坐标（考虑面积权重）
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        # 退化情况（如所有点共线或单点）：回退到几何平均
        cx, cy = np.mean(contour[:, 0, :], axis=0)
    
    return cx, cy


def ana_geojson_ann(ann_file):

    tp_name_id_map = {
        'nuclei_tumor': 0,
        'nuclei_lymphocyte': 1,
        'nuclei_plasma_cell': 2,
        'nuclei_histiocyte': 3,
        'nuclei_melanophage': 4,
        'nuclei_neutrophil': 5,
        'nuclei_stroma': 6,
        'nuclei_endothelium': 7,
        'nuclei_epithelium': 8,
        'nuclei_apoptosis': 9
    }

    ann_data = json.load(open(ann_file))['features']

    cells = list()
    for cell_info in ann_data:
        geo = cell_info['geometry']['coordinates']
        tp = cell_info['properties']['classification']['name']
        _id = cell_info['id']

        for true_poly in geo:
            try:
                cx, cy = compute_center(true_poly)
                cells.append([cx, cy, tp_name_id_map[tp]])
            except:
                print(f'Something wrong, may because of the poly includes multiple contour ({len(true_poly)}), try to analyze it')
                for _poly in true_poly:
                    cx, cy = compute_center(_poly)
                    cells.append([cx, cy, tp_name_id_map[tp]])
    
    return np.array(cells)


if __name__ == '__main__':

    # ========================
    # build the metadata
    # ========================
    print('======== build the metadata =======')
    nuclei_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_geojson_nuclei')
    central_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_tif_ROIs')
    context_ann_folder = os.path.join(args.puma_folder, '01_training_dataset_tif_context_ROIs')

    samples = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(central_ann_folder + '/*.tif')]
    print(f'Fine {len(samples)} samples')

    print('======== Ana all nuclei info =======')
    cell_ann_dict = dict()
    for sample_name in samples:
        ann_file = os.path.join(nuclei_ann_folder, sample_name + '_nuclei' + '.geojson')
        cell_ann_dict[sample_name] = ana_geojson_ann(ann_file)
    
    print('======== Load the split info =======')
    split_meta = dict()
    split_meta = ana_split_file(split_meta, os.path.join(args.puma_folder, 'split', 'train.json'), 'train')
    split_meta = ana_split_file(split_meta, os.path.join(args.puma_folder, 'split', 'val.json'), 'val')
    split_meta = ana_split_file(split_meta, os.path.join(args.puma_folder, 'split', 'test.json'), 'test')
    print(split_meta)
    
    # ========================
    # loop all samples
    # ========================
    save_dict = dict()
    for raw_sample_name in tqdm.tqdm(samples):

        raw_crop_x = (5120 - 1024) // 2
        raw_crop_y = (5120 - 1024) // 2
        crop_size = 1024

        context_image = cv2.imread(os.path.join(context_ann_folder, f'{raw_sample_name}_context.tif'))

        ann = cell_ann_dict[raw_sample_name]

        split = split_meta[raw_sample_name]

        wsi_mpp = 0.25  # we suppose the default is 40x
        for target_mpp in [0.25, 0.50]:
            target_central_size = int(224 * (target_mpp / wsi_mpp))
            target_patch_sizes = [int(224 * (target_mpp / wsi_mpp)), int(512 * (target_mpp / wsi_mpp)), int(1024 * (target_mpp / wsi_mpp))]
            save_patch_sizes = [224, 512, 1024]
            # crop the annotated sample based on the target size (no overlap)
            x_num = int(np.ceil(1024 / target_central_size))

            for x_id in range(x_num):
                for y_id in range(x_num):
                    shift_x = x_id * target_central_size  # local coord (raw ann patch)
                    shift_y = y_id * target_central_size  # local coord (raw ann patch)
                    crop_x = raw_crop_x + shift_x  # global coord (wsi)
                    crop_y = raw_crop_y + shift_y  # global coord (wsi)

                    # crop the annotation
                    ann_crop = ann[
                        (ann[:, 0] >= shift_x) & (ann[:, 0] < shift_x + target_central_size) &
                        (ann[:, 1] >= shift_y) & (ann[:, 1] < shift_y + target_central_size)
                    ]

                    # we skip the patches that have no annotations
                    if len(ann_crop) == 0:
                        continue

                    ann_crop[:, :2] = ann_crop[:, :2] - np.array([shift_x, shift_y])  # local coord (central patch)
                    ann_crop[:, :2] = ann_crop[:, :2] + (target_patch_sizes[-1] - target_central_size) // 2

                    # crop the max patch from wsi
                    crop_x = crop_x - (target_patch_sizes[-1] - target_central_size) // 2  # global coord (wsi)
                    crop_y = crop_y - (target_patch_sizes[-1] - target_central_size) // 2  # global coord (wsi)

                    wsi_crop = context_image[crop_y:crop_y + target_patch_sizes[-1], crop_x:crop_x + target_patch_sizes[-1]]

                    # build the true sample
                    for i, target_patch_size in enumerate(target_patch_sizes):

                        sample_name = f'{split}_{raw_sample_name}_{x_id}_{y_id}'

                        # center crop
                        true_patch = wsi_crop[
                            (target_patch_sizes[-1] - target_patch_size) // 2 : (target_patch_sizes[-1] - target_patch_size) // 2 + target_patch_size,
                            (target_patch_sizes[-1] - target_patch_size) // 2 : (target_patch_sizes[-1] - target_patch_size) // 2 + target_patch_size
                        ]
                        # shift the ann_crop
                        true_ann = ann_crop.copy()
                        true_ann[:, :2] = true_ann[:, :2] - (target_patch_sizes[-1] - target_patch_size) // 2

                        true_size = save_patch_sizes[i]
                        true_scale_ratio = true_size / target_patch_size
                        
                        true_ann[:, :2] = true_ann[:, :2] * true_scale_ratio
                        true_ann = np.array(true_ann, dtype=int)
                        true_patch = cv2.resize(true_patch, (true_size, true_size), interpolation=cv2.INTER_CUBIC)
                        true_patch = cv2.cvtColor(true_patch, cv2.COLOR_BGR2RGB)
                        
                        for ann_type in ['coarse']:
                            
                            group_name = f'mpp_{target_mpp}-size_{save_patch_sizes[i]}-label_{ann_type}-valid_224'
                            
                            if ann_type == 'coarse':
                                save_ann = true_ann.copy()
                                # map the label > 2 to 2
                                save_ann[:, 2][save_ann[:, 2] > 2] = 2
                            else:
                                save_ann = true_ann.copy()

                            # save to the dict
                            if group_name not in save_dict:
                                save_dict[group_name] = {
                                    'patches': dict(),
                                    'anns': dict()
                                }
                            save_dict[group_name]['patches'][sample_name] = np.array(true_patch, dtype=np.uint8)

                            # for linear / knn eval
                            # the bg is ignored
                            save_dict[group_name]['anns'][sample_name] = np.array(save_ann, dtype=np.int32)

                            # the ann should be in the true_patch
                            # otherwise, there may be some errors
                            assert np.all(save_ann[:, :2] >= 0) and np.all(save_ann[:, :2] < true_size), f'{raw_img_name} {x_id} {y_id}: the ann is out of the patch, please check it'

                            # draw the ann
                            preview_save_dir = os.path.join(args.out_folder, f'{group_name}_preview')
                            os.makedirs(preview_save_dir, exist_ok=True)
                            draw_img = np.array(true_patch)
                            for x, y, t in save_ann:
                                x = int(x)
                                y = int(y)
                                if t == 0:
                                    cv2.circle(draw_img, (x, y), 5, (0, 162, 232), -1)
                                elif t == 1:
                                    cv2.circle(draw_img, (x, y), 5, (255, 0, 0), -1)
                                elif t == 2:
                                    cv2.circle(draw_img, (x, y), 5, (0, 255, 0), -1)
                                elif t == 3:
                                    cv2.circle(draw_img, (x, y), 5, (0, 0, 255), -1)
                                elif t == 4:
                                    cv2.circle(draw_img, (x, y), 5, (255, 255, 0), -1)
                                elif t == 5:
                                    cv2.circle(draw_img, (x, y), 5, (255, 0, 255), -1)
                                elif t == 6:
                                    cv2.circle(draw_img, (x, y), 5, (0, 255, 255), -1)
                                elif t == 7:
                                    cv2.circle(draw_img, (x, y), 5, (122, 0, 255), -1)
                                elif t == 8:
                                    cv2.circle(draw_img, (x, y), 5, (0, 255, 122), -1)
                                elif t == 9:
                                    cv2.circle(draw_img, (x, y), 5, (255, 0, 122), -1)
                            
                            draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(os.path.join(preview_save_dir, f'{sample_name}.png'), draw_img)
    
    # save the save_dict
    for group_name in save_dict.keys():
        np.save(f'{args.out_folder}/{group_name}.npy', save_dict[group_name])
