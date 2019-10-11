from scipy import io
import os
import cv2
from tqdm import tqdm
import numpy as np


def load_save_first_last(save_path="/data/proj/registration/result/check_first_last"):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for vol_num in tqdm(range(201, 202)):
        if vol_num in [6, 97, 125, 198, 199, 228, 271]:
            continue
        data = io.loadmat("/data/dataset/urinary/vol_{}.mat".format(vol_num))
        pre = data['imgV_pre']
        delay = data['imgV_delay']

        os.mkdir(os.path.join(save_path, 'vol_{}'.format(vol_num)))

        # save pre first
        cv2.imwrite(os.path.join(save_path, "vol_{}".format(vol_num), "0_pre.png"),
                    (pre[:, :, 0] / 4096 * 255).astype(np.uint8))

        # save pre last
        cv2.imwrite(os.path.join(save_path, "vol_{}".format(vol_num), "{}_pre.png".format(pre.shape[-1])),
                    (pre[:, :, -1] / 4096 * 255).astype(np.uint8))

        # save delay first
        cv2.imwrite(os.path.join(save_path, "vol_{}".format(vol_num), "0_delay.png"),
                    (delay[:, :, 0] / 4096 * 255).astype(np.uint8))

        # save delay last
        cv2.imwrite(os.path.join(save_path, "vol_{}".format(vol_num), "{}_delay.png".format(delay.shape[-1])),
                    (delay[:, :, -1] / 4096 * 255).astype(np.uint8))

        del data, pre, delay


if __name__ == "__main__":
    load_save_first_last()
