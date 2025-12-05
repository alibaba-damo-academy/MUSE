"""
This file is used to build the bit stream samples for SSL.
"""
import os
import shutil
import h5py
from PIL import Image
import io
import numpy as np
import argparse
import pandas as pd
from multiprocessing import cpu_count, Pool


def build_byte_stream_samples(input_h5, output_folder):
    os.makedirs(os.path.join(output_folder, 'samples'), exist_ok=True)
    output_h5 = os.path.join(output_folder, 'samples', os.path.basename(input_h5))

    with h5py.File(input_h5, 'r') as f:
        with h5py.File(output_h5, 'w') as f_out:

            # dt = h5py.vlen_dtype(np.dtype('uint8'))
            # dset = f_out.create_dataset('patch', shape=(1,), dtype=dt)
            patch = Image.fromarray(f['patch'][:])
            with io.BytesIO() as buffer:
                patch.save(buffer, format='JPEG')
                patch_bytes = buffer.getvalue()
            f_out.create_dataset(
                'patch',
                data=np.frombuffer(patch_bytes, dtype='uint8'), 
                dtype=np.uint8
            )

            f_out.create_dataset('cell', data=f['cell'][:])
    
    print(f'save {output_h5}')


def decoder_byte_stream_samples(input_h5):

    with h5py.File(input_h5, 'r') as f:
        patch = Image.open(io.BytesIO(f['patch'][:].tobytes()))

        print(patch.size, f['cell'].shape)
        
    return patch
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_folder', type=str, required=True)
    parser.add_argument('--output_data_folder', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_data_folder, exist_ok=True)

    # 1. copy the args.txt to output_folder
    shutil.copy(os.path.join(args.input_data_folder, 'args.txt'), args.output_data_folder)

    # 2. copy the metadata.csv to output_folder
    shutil.copy(os.path.join(args.input_data_folder, 'metadata.csv'), args.output_data_folder)

    # 3. build tasks
    metadata = pd.read_csv(os.path.join(args.input_data_folder, 'metadata.csv'))

    tasks = list()
    for index, row in metadata.iterrows():
        sample_name = row['sample_name']

        h5_file = os.path.join(args.input_data_folder, 'samples', sample_name + '.h5')

        tasks.append([h5_file, args.output_data_folder])
    
    # 4. run
    with Pool(cpu_count()) as p:
        p.starmap(build_byte_stream_samples, tasks)
