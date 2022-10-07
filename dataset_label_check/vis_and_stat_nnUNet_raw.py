import os, sys, pdb
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from petrel_client.client import Client
import io
import gzip
from nibabel import Nifti1Image, FileHolder
from scipy import ndimage as ndi
from tqdm import tqdm

def load_niigz_from_ceph(client, url):
    ''' nib data shape: [x, y, z]
    '''
    with io.BytesIO(client.get(url)) as f:
        f = gzip.open(f)
        fh = FileHolder(fileobj=f)
        nib_data = Nifti1Image.from_file_map({'header': fh, 'image':fh})
        nib_data = Nifti1Image.from_bytes(nib_data.to_bytes())
    return nib_data

def load_json_from_ceph(client, url):
    try:
        data = json.loads(client.get(url, update_cache=True))
    except:
        print("NOT FOUND", url)
    return data

def is_ceph_file(client, url):
    return client.contains(url)

def generate_dataset_statistic(client, json_dataset, inp_root):
    
    jso = json_dataset['training']
    target_dict = {}
    vis_dict = {'max_spacing': [None] * 3, 'min_spacing': [None] * 3,
                'max_phy_size': [None] * 3, 'min_phy_size': [None] * 3}
    max_spacing_list = [-1, -1, -1]
    min_spacing_list = [1e5, 1e5, 1e5]
    max_phy_size_list = [-1, -1, -1]
    min_phy_size_list = [1e5, 1e5, 1e5]
    for j, item in tqdm(enumerate(jso)):
    # for item in jso:
        file_name = item['image']
        if file_name.startswith('./'):
            file_name = file_name[2:]
        assert file_name.startswith('imagesTr')

        file_url = os.path.join(inp_root, file_name)
        if not is_ceph_file(client, file_url):
            file_url = file_url.replace('.nii.gz', '_0000.nii.gz')
        assert is_ceph_file(client, file_url)

        nib_data = load_niigz_from_ceph(client, file_url)
        spacing = nib_data.header['pixdim'][1:4] # x, y, z
        size = nib_data.shape # x, y, z

        target_dict[file_name] = {}
        target_dict[file_name]['size'] = size
        target_dict[file_name]['spacing'] = spacing

        phy_size = np.array(spacing) * np.array(size)[:3]

        for i in range(len(spacing)):
            if spacing[i] > max_spacing_list[i]:
                vis_dict['max_spacing'][i] = file_url
                max_spacing_list[i] = spacing[i]
            if spacing[i] < min_spacing_list[i]:
                vis_dict['min_spacing'][i] = file_url
                min_spacing_list[i] = spacing[i]
            
            if phy_size[i] > max_phy_size_list[i]:
                vis_dict['max_phy_size'][i] = file_url
                max_phy_size_list[i] = phy_size[i]
            if phy_size[i] < min_phy_size_list[i]:
                vis_dict['min_phy_size'][i] = file_url
                min_phy_size_list[i] = phy_size[i]

    df_stat = pd.DataFrame.from_dict(target_dict, orient='index')
    return df_stat, vis_dict

def visualize_dataset(client, vis_dict, vis_root):
    
    co_dict = {0: 'x', 1: 'y', 2: 'z'}
    for k, v in vis_dict.items():
        for i, file_url in enumerate(v):
            nib_data = load_niigz_from_ceph(client, file_url)
            
            spacing = nib_data.header['pixdim'][1:4] # x, y, z
            size = nib_data.shape # x, y, z
            nib_arr = nib_data.get_fdata()
            assert nib_arr.shape == size

            # For better display experience. It is based on axis x.
            # You can comment it for accelerate speed...
            zoom_factors = (1, spacing[1] / spacing[0], spacing[2] / spacing[0])
            nib_arr = ndi.zoom(nib_arr, zoom_factors)

            x, y, z = nib_arr.shape
            x_slide = nib_arr[x//2, :, :]
            y_slide = nib_arr[:, y//2, :]
            z_slide = nib_arr[:, :, z//2]


            sub_vis_root = os.path.join(vis_root, '{}_{}_{}_{}'.format(k, co_dict[i], round(spacing[i], 2), int(spacing[i] * size[i])))
            os.makedirs(sub_vis_root, exist_ok=True)
            
            save_file = '{}_{}.png'.format # i.e. x_xxxxx.nii.gz.png
            plt.imsave(os.path.join(sub_vis_root, save_file('x', file_url.split('/')[-1])), x_slide)
            plt.imsave(os.path.join(sub_vis_root, save_file('y', file_url.split('/')[-1])), y_slide)
            plt.imsave(os.path.join(sub_vis_root, save_file('z', file_url.split('/')[-1])), z_slide)
            # plt.imsave(os.path.join(vis_root, save_file(k, 'x', '_'.join(map(str, spacing)), file_url.split('/')[-1])), x_slide)
            # plt.imsave(os.path.join(vis_root, save_file(k, 'y', '_'.join(map(str, spacing)), file_url.split('/')[-1])), y_slide)
            # plt.imsave(os.path.join(vis_root, save_file(k, 'z', '_'.join(map(str, spacing)), file_url.split('/')[-1])), z_slide)


if __name__ == "__main__":
    inp_root, save_root, vis_root = sys.argv[1:]
    # inp_root = 's3://nnUNet_raw_data/Task941_GLIS-RT/'
    # save_root = 'stats_results/Task941_GLIS-RT/'
    # vis_root = 'visualization/Task941_GLIS-RT/'

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)

    client_why = Client('~/petreloss_why.conf')
    # client_yj = Client('~/petreloss.conf')

    json_dataset = load_json_from_ceph(client_why, os.path.join(inp_root, 'dataset.json'))

    print ('* Now generating statistic information of dataset: {}'.format(inp_root))
    if(not os.path.exists(os.path.join(save_root, 'dataset_statistic.csv'))):
        df_stat, vis_dict = generate_dataset_statistic(client_why, json_dataset, inp_root)
        df_stat.to_csv(os.path.join(save_root, 'dataset_statistic.csv'))

    print ('* Now visualizing some cases of dataset: {}'.format(inp_root))
    visualize_dataset(client_why, vis_dict, vis_root)
