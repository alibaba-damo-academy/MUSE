import os
import tqdm
import json
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser("Restore necessary WSIs for OCELOT dataset")
parser.add_argument("--wsi_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, default='../data/wsi/ocelot')
parser.add_argument("--ocelot_meta_path", type=str, default='../data/OCELOT/metadata.json')
parser.add_argument("--tcga_meta_path", type=str, default='../data/metadata/GDC_metadata.csv')
args = parser.parse_args()


if __name__ == '__main__':

    # ====================
    # Metadata
    # ====================
    with open(args.ocelot_meta_path, 'r') as f:
        meta = json.load(f)

    samples_meta = meta['sample_pairs']

    # ====================
    # Get unique WSIs
    # ====================
    # for each wsi, there are two important information
    # 1. slide_name
    # 2. organ
    wsi_info_list = list()

    tcga_meta = pd.read_csv(args.tcga_meta_path)

    for key, value in samples_meta.items():

        # find the entity_id
        tmp_row = tcga_meta[tcga_meta['File Name'] == value['slide_name']+'.svs']
        assert len(tmp_row) == 1, '{} is not in the TCGA meta'.format(value['slide_name'])
        file_id = tmp_row['file_id'].values[0]

        # build the tuple
        wsi_info = (file_id, value['slide_name'], value['organ'])

        # check if the tuple is already in the list
        if wsi_info not in wsi_info_list:
            wsi_info_list.append(wsi_info)

    print('Find {} samples, {} unique WSIs'.format(len(samples_meta), len(wsi_info_list)))

    # ====================
    # Restore WSIs
    # ====================
    os.makedirs(args.output_folder, exist_ok=True)
    for meta_tuple in tqdm.tqdm(wsi_info_list):
        file_id = meta_tuple[0]
        slide_name = meta_tuple[1]
        organ = meta_tuple[2]

        wsi_path = os.path.join(args.wsi_folder, file_id, slide_name+'.svs')

        # check existence
        assert os.path.exists(wsi_path), '{} does not exist'.format(wsi_path)

        # copy to output folder
        output_path = os.path.join(args.output_folder, slide_name+'.svs')
        shutil.copyfile(wsi_path, output_path)
