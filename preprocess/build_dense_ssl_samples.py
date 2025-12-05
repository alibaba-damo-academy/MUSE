import os
import cv2
import glob
import h5py
import tqdm
import argparse
import openslide
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_folder', required=True, help='Path to the folder containing the whole slide images')
parser.add_argument('--patch_meta_folder', required=True, help='Path to the folder containing the patch meta files')
parser.add_argument('--cell_folder', required=True, help='Path to the folder containing the cell')
parser.add_argument('--output_folder', required=True, help='Path to the folder where the dense ssl samples will be saved')
parser.add_argument('--target_size', type=int, default=2048, help='Patch size')
parser.add_argument('--target_mpp', type=float, default=0.25, help='Target mpp')
args = parser.parse_args()


def restore_valid_patches(wsi_file, patch_meta_file, cell_file, output_folder, target_size, target_mpp):

    true_output_folder = os.path.join(output_folder, 'samples')
    preview_folder = os.path.join(output_folder, 'preview')
    os.makedirs(true_output_folder, exist_ok=True)
    os.makedirs(preview_folder, exist_ok=True)

    wsi = openslide.OpenSlide(wsi_file)
    cells = h5py.File(cell_file, 'r')['prediction'][:, :2]
    patch_meta = h5py.File(patch_meta_file, 'r')['coords'][:]

    if 'aperio.MPP' in wsi.properties:
        raw_mpp = float(wsi.properties['aperio.MPP'])
    else:
        raw_mpp = 0.25
    
    crop_patch_size = int(target_size * target_mpp / raw_mpp)

    wsi_name = os.path.splitext(os.path.basename(wsi_file))[0]

    for patch_id, patch_coord in enumerate(patch_meta):

        # init output file
        output_file = os.path.join(true_output_folder, wsi_name + f'_{patch_id}.h5')
        if os.path.exists(output_file):
            print('{} already exists'.format(output_file))
            continue
        else:
            print('Processing {}'.format(output_file))

        # init cell data 
        begin_x, begin_y = patch_coord
        center_x = begin_x + crop_patch_size / 2
        center_y = begin_y + crop_patch_size / 2
        cropped_cells = cells[np.logical_and(
            abs(cells[:, 0] - center_x) < crop_patch_size / 2, 
            abs(cells[:, 1] - center_y) < crop_patch_size / 2
        )]
        if cropped_cells.shape[0] > 0:
            cropped_cells = cropped_cells - np.array([begin_x, begin_y])
            cropped_cells = cropped_cells * raw_mpp / target_mpp

        # init image
        cropped_patch = np.array(wsi.read_region(
            location=(begin_x, begin_y),
            level=0,
            size=(crop_patch_size, crop_patch_size)
        ))
        cropped_patch = cv2.resize(cropped_patch, (target_size, target_size))[:, :, :3]

        # save
        with h5py.File(output_file, 'w') as f:

            # valid patch
            f.create_dataset(
                f'patch',
                data=cropped_patch
            )

            f.create_dataset(
                f'cell',
                data=cropped_cells
            )

        # build preview
        # resize to 512
        show_patch = cv2.resize(cropped_patch, (512, 512))
        show_cells = cropped_cells.copy() / (target_size / 512)
        for cell_info in show_cells:
            cv2.circle(show_patch, (int(cell_info[0]), int(cell_info[1])), 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(preview_folder, f'{wsi_name}_{patch_id}.png'), show_patch[:, :, ::-1])


if __name__ == '__main__':

    os.makedirs(args.output_folder, exist_ok=True)

    wsi_files = glob.glob(os.path.join(args.wsi_folder, '*.svs'))

    # build tasks
    tasks = list()
    for wsi_file in tqdm.tqdm(wsi_files):
        sample_name = os.path.splitext(os.path.basename(wsi_file))[0]

        patch_meta_file = os.path.join(args.patch_meta_folder, sample_name + '.h5')
        cell_file = os.path.join(args.cell_folder, sample_name + '.h5')

        # check exists
        if os.path.exists(patch_meta_file) and os.path.exists(cell_file):
            tasks.append((wsi_file, patch_meta_file, cell_file, args.output_folder, args.target_size, args.target_mpp))
    
    print('Total number of tasks: {}'.format(len(tasks)))

    # run
    with Pool(cpu_count()) as p:
        p.starmap(restore_valid_patches, tasks)

    # build metadata
    meta_dict = {
        'sample_name': list(),
        'wsi_name': list(),
        'patch_id': list(),
        'n_cell': list(),
    }

    samples = glob.glob(os.path.join(args.output_folder, 'samples', '*.h5'))
    for sample in tqdm.tqdm(samples):
        sample_name = os.path.splitext(os.path.basename(sample))[0]
        wsi_name = '_'.join(sample_name.split('_')[:-1])
        patch_id = eval(sample_name.split('_')[-1])
        meta_dict['sample_name'].append(sample_name)
        meta_dict['wsi_name'].append(wsi_name)
        meta_dict['patch_id'].append(patch_id)
        with h5py.File(sample, 'r') as f:
            meta_dict['n_cell'].append(f['cell'].shape[0])
    
    meta_df = pd.DataFrame(meta_dict)
    meta_df.to_csv(os.path.join(args.output_folder, 'metadata.csv'), index=False)
