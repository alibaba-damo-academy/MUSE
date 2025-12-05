import os
import glob
import tqdm
import json
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser("Restore necessary WSIs for BRCAM2C dataset")
parser.add_argument("--wsi_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--brcam2c_patch_folder", type=str, required=True)
parser.add_argument("--tcga_meta_path", type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':

    # ====================
    # analysis patch folder
    # ====================
    sample_ids = list()
    for patch_path in glob.glob(args.brcam2c_patch_folder + '/*'):
        patch_name = os.path.splitext(os.path.basename(patch_path))[0]
        sample_ids.append(patch_name.split('_')[0])

    # ====================
    # Get unique WSIs
    # ====================
    # for each wsi, there are two important information
    # 1. slide_name
    # 2. organ
    wsi_info_list = list()

    tcga_meta = pd.read_csv(args.tcga_meta_path)
    # build a new column,that split the case id from file_name
    tcga_meta['sample_id'] = tcga_meta['file_name'].apply(lambda x: x.split('.')[0]).copy()

    for sample_id in sample_ids:

        tmp_row = tcga_meta[tcga_meta['sample_id'] == sample_id]
        assert len(tmp_row) == 1, 'Find {} row for {}'.format(len(tmp_row), sample_id)

        file_id = tmp_row['file_id'].values[0]
        file_name = tmp_row['file_name'].values[0]

        # build the tuple
        wsi_info = (file_id, file_name)

        # check if the tuple is already in the list
        if wsi_info not in wsi_info_list:
            wsi_info_list.append(wsi_info)

    print('Find {} samples, {} unique WSIs'.format(len(sample_ids), len(wsi_info_list)))

    # ====================
    # Restore WSIs
    # ====================
    os.makedirs(args.output_folder, exist_ok=True)
    for meta_tuple in tqdm.tqdm(wsi_info_list):
        file_id = meta_tuple[0]
        slide_name = meta_tuple[1]

        wsi_path = os.path.join(args.wsi_folder, file_id, slide_name)

        # check existence
        assert os.path.exists(wsi_path), '{} does not exist'.format(wsi_path)

        # copy to output folder
        output_path = os.path.join(args.output_folder, slide_name)
        shutil.copyfile(wsi_path, output_path)
