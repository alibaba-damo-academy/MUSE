"""
This file is used to build the data for linear/knn evaluation

args:
- brcam2c_folder: the folder of brcam2c
- wsi_folder: the folder of WSIs
- out_folder: the folder to save the data

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
import tqdm
import argparse
import openslide
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--brcam2c_folder', type=str, default='', help='the folder of brcam2c')
parser.add_argument('--wsi_folder', type=str, default='', help='the folder of WSIs')
parser.add_argument('--out_folder', type=str, default='', help='the folder to save the data')
args = parser.parse_args()


def analysis_split_file(split_file, record_dict, split):

    for line in open(split_file):
        sample_info = line.strip()
        sample_name = sample_info.split('_')[0]
        crop_info = sample_info.split('_')[1:3]
        crop_x, crop_y = [int(x) for x in crop_info]

        wsi_name = glob.glob(f'{args.wsi_folder}/{sample_name}*')[0]
        wsi_name = os.path.splitext(os.path.basename(wsi_name))[0]

        record_dict['raw_img_name'].append(os.path.splitext(sample_info)[0])
        record_dict['sample_name'].append(sample_name)
        record_dict['wsi_name'].append(wsi_name)
        record_dict['split'].append(split)
        record_dict['crop_x'].append(crop_x)
        record_dict['crop_y'].append(crop_y)
        record_dict['crop_size'].append(1000)
    
    return record_dict


if __name__ == '__main__':

    # ========================
    # build the metadata
    # ========================
    print('======== build the metadata =======')
    meta_dict = {
        'raw_img_name': list(),
        'sample_name': list(),
        'wsi_name': list(),
        'split': list(),
        'crop_x': list(),
        'crop_y': list(),
        'crop_size': list()
    }

    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_train.txt'), meta_dict, 'train')
    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_val.txt'), meta_dict, 'val')
    meta_dict = analysis_split_file(os.path.join(args.brcam2c_folder, 'brca_ds_test.txt'), meta_dict, 'test')

    meta_df = pd.DataFrame(meta_dict).reset_index()

    print(meta_df)

    # ========================
    # loop all samples
    # ========================
    save_dict = dict()
    for row in tqdm.tqdm(meta_df.iterrows()):

        # =========================
        # get the basic information
        # =========================
        raw_img_name = row[1]['raw_img_name']
        split = row[1]['split']
        wsi_name = row[1]['wsi_name']
        raw_crop_x = row[1]['crop_x']
        raw_crop_y = row[1]['crop_y']
        crop_size = 1000  # setting the crop_size == 1000 is ok.

        wsi = openslide.OpenSlide(f'{args.wsi_folder}/{wsi_name}.svs')
        wsi_mpp = eval(wsi.properties['aperio.MPP'])

        # for brcam2c, we should map the coordinates to the raw mpp
        ann_mpp = 0.5
        ann = np.loadtxt(
            f'{args.brcam2c_folder}/labels/{raw_img_name}_gt_class_coords.txt',
            dtype=int,
            delimiter=' '
        )
        ann[:, :2] = ann[:, :2] * (ann_mpp / wsi_mpp)
        ann = ann[:, [1, 0, 2]]  # the order of the columns is x, y, w

        # ===========================
        # split the annotated sample
        # ===========================
        for target_mpp in [0.25, 0.50]:
            target_central_size = int(224 * (target_mpp / wsi_mpp))
            target_patch_sizes = [int(224 * (target_mpp / wsi_mpp)), int(512 * (target_mpp / wsi_mpp)), int(1024 * (target_mpp / wsi_mpp))]
            save_patch_sizes = [224, 512, 1024]
            # crop the annotated sample based on the target size (no overlap)
            x_num = int(np.ceil(1000 / target_central_size))

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

                    wsi_crop = np.array(wsi.read_region(
                        (crop_x, crop_y),
                        0,
                        (target_patch_sizes[-1], target_patch_sizes[-1])
                    ).convert('RGB'), dtype=np.uint8)

                    # build the true sample
                    for i, target_patch_size in enumerate(target_patch_sizes):
                        
                        group_name = f'mpp_{target_mpp}-size_{save_patch_sizes[i]}-valid_224'
                        sample_name = f'{split}_{raw_img_name}_{x_id}_{y_id}'

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

                        # save to the dict
                        if group_name not in save_dict:
                            save_dict[group_name] = {
                                'patches': dict(),
                                'anns': dict()
                            }
                        save_dict[group_name]['patches'][sample_name] = np.array(true_patch, dtype=np.uint8)

                        # for linear / knn eval
                        # the bg is ignored
                        # therefore, the ann should - 1
                        true_ann[:, 2] -= 1
                        save_dict[group_name]['anns'][sample_name] = np.array(true_ann, dtype=np.int32)

                        # the ann should be in the true_patch
                        # otherwise, there may be some errors
                        assert np.all(true_ann[:, :2] >= 0) and np.all(true_ann[:, :2] < true_size), f'{raw_img_name} {x_id} {y_id}: the ann is out of the patch, please check it'

                        # draw the ann
                        preview_save_dir = os.path.join(args.out_folder, f'{group_name}_preview')
                        os.makedirs(preview_save_dir, exist_ok=True)
                        draw_img = np.array(true_patch)
                        for x, y, t in true_ann:
                            x = int(x)
                            y = int(y)
                            if t == 0:
                                cv2.circle(draw_img, (x, y), 5, (0, 162, 232), -1)
                            elif t == 1:
                                cv2.circle(draw_img, (x, y), 5, (255, 0, 0), -1)
                            elif t == 2:
                                cv2.circle(draw_img, (x, y), 5, (255, 255, 0), -1)
                        
                        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(preview_save_dir, f'{sample_name}.png'), draw_img)
    
    # save the save_dict
    for group_name in save_dict.keys():
        np.save(f'{args.out_folder}/{group_name}.npy', save_dict[group_name])
