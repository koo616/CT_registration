import os
import nibabel as nib
import numpy as np

# params
path = 'result'
expr_name = 'align200'
epoch = 42
data_dir = os.path.join(path, "{}_epoch{}".format(expr_name, epoch))
vol_num = 1

# load data
delay = nib.load(os.path.join(data_dir, 'vol_{}_reference.nii.gz'.format(vol_num)))
regist = nib.load(os.path.join(data_dir, 'vol_{}_registration.nii.gz'.format(vol_num)))

# nii to numpy
delay = delay.get_fdata()
regist = regist.get_fdata()

# partition
delay_part = delay[80:-40, 50:-50, :]
regist_part = regist[80:-40, 50:-50, :]

# stat
diff = delay_part - regist_part  # delay - pre -> to get the part brighter than pre img
# get mean and std without background
diff_mean = diff[(regist_part > 0.20) & (delay_part > 0.20)].mean()
diff_std = diff[(regist_part > 0.20) & (delay_part > 0.20)].std()
# standardization
diff_standard = (diff - diff_mean) / diff_std
# check quantile
np.quantile(diff_standard, q=0.9999)

# the part which is brighter than pre img, not containing fully dark part
diff_segmap = ((diff_standard > 17) & (regist_part > 0.15)).astype(int)
diff_segmap = diff_segmap.astype(float)
# check sum
diff_segmap.sum()


# padding
segmap = np.zeros(shape=delay.shape)
segmap[80:-40, 50:-50, :] = diff_segmap

# save
segmap_img = nib.Nifti1Image(segmap, affine=np.eye(4))
nib.save(segmap_img, os.path.join(data_dir, "vol_1_segmap.nii.gz"))