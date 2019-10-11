import os
from scipy import io
from tqdm import tqdm
import nibabel as nib
import numpy as np


def mat_to_nib(vol_names, save_path):
    for vol_name in tqdm(vol_names):
        vol = io.loadmat(os.path.join("/data/dataset/urinary", vol_name))
        pre = vol['imgV_pre']
        delay = vol['imgV_delay']

        pre = np.flip(pre, 2)
        delay = np.flip(delay, 2)

        pre = np.rot90(pre, 3)
        delay = np.rot90(delay, 3)

        if not os.path.isdir(os.path.join(save_path, vol_name.split('.')[0])):
            os.mkdir(os.path.join(save_path, vol_name.split('.')[0]))

        warp_img = nib.Nifti1Image(pre, affine=np.eye(4))
        nib.save(warp_img, os.path.join(save_path, vol_name.split('.')[0], 'pre.nii.gz'))

        warp_img = nib.Nifti1Image(delay, affine=np.eye(4))
        nib.save(warp_img, os.path.join(save_path, vol_name.split('.')[0], 'delay.nii.gz'))


if __name__ == '__main__':
    data_dir = "/data/dataset/urinary"
    save_path = "../result/showing"

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    vol_names = [filename for filename in os.listdir(data_dir)
                 if int(filename.split("_")[-1].split(".")[0]) in (9, 130, 128)]
    mat_to_nib(vol_names, save_path)
