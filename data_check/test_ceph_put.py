
import io
import numpy as np
import pickle
from petrel_client.client import Client
import os.path as osp


def save_npy_to_ceph(client, fname, data): 
    with io.BytesIO() as f:
        np.save(f, data)
        f.seek(0)
        client.put('s3://{}'.format(fname), f)

def save_pkl_to_ceph(client, fname, data): 
    with io.BytesIO() as f:
        pickle.dump(data, f)
        f.seek(0)
        client.put('s3://{}'.format(fname), f)

import struct
import ast

def read_data_from_ceph(client, img_url):
    # img_url = 's3://test_bucket/autoPET_0001.npy'
    img_bytes = client.get(img_url)
    assert(img_bytes is not None)
    img_mem_view = memoryview(img_bytes)
    assert(img_mem_view[:6] ==  b'\x93NUMPY') # The first 6 bytes are a magic string: exactly \x93NUMPY.
    assert(img_mem_view[6:8] == b'\x01\x00') # The next 2 bytes are version of npy format. Must be 1.0

    len_header = struct.unpack("<h", img_mem_view[8:10]) # must be 118
    assert (len(len_header) == 1)
    len_header = len_header[0]

    header = img_bytes[10:10+len_header] # must be img_bytes...
    header_info = ast.literal_eval(header.decode("utf-8"))

    case_all_data = np.frombuffer(img_mem_view[10+len_header:], header_info['descr']).reshape(header_info['shape'])
    return case_all_data

def load_numpy_from_ceph(client, url):
    ''' Any numpy file can be load, such as 'npz', 'npy'
    '''
    data_bytes = client.get(url, enable_cache=True)
    print(url)
    # try:
    data = np.load(io.BytesIO(data_bytes))
    # except:
        # import pdb; pdb.set_trace()
    return data

if __name__ == "__main__":
    # ceph_folder = "nnUNet_predict_data/"
    # client = Client(enable_mc=True, conf_path="~/why_ceph.conf")
    client = Client(enable_mc=True, conf_path="~/petreloss.conf")
    # path = "s3://nnUNet_preprocessed/Task020_AbdomenCT1K/nnUNetData_plans_General_stage0/Case_00541.npy"
    path = "s3://nnUNet_preprocessed/Task083_VerSe2020/nnUNetData_plans_General_stage1/sub-verse711.npy"
    data = load_numpy_from_ceph(client, path)
    # data = read_data_from_ceph(client, path)
    # print(osp.join(ceph_folder, "test_arr.pkl"))
    # data = read_data_from_ceph(client, osp.join("s3://", ceph_folder, "REcmg3_0.3amos_0368.npy"))
    # save_pkl_to_ceph(client, osp.join(ceph_folder, "test_arr.pkl"), data)

    # ceph_folder = "test_bucket_why/en_test/"
    # client = Client(enable_mc=True, conf_path="~/why_ceph.conf")
    # data = np.ones((3,3))
    # print(osp.join(ceph_folder, "test_arr.npy"))
    # # print(osp.join(ceph_folder, "test_arr.pkl"))
    # save_npy_to_ceph(client, osp.join(ceph_folder, "test_arr.npy"), data)
    # # save_pkl_to_ceph(client, osp.join(ceph_folder, "test_arr.pkl"), data)

    print("end")
