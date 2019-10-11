# py imports
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm
from scipy import io
import cv2

# third party
import tensorflow as tf
import numpy as np
import nibabel as nib
from keras.backend.tensorflow_backend import set_session
from skimage import measure

# project
sys.path.append('../ext/medipy-lib')
import networks

normal = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 27, 29, 31, 32, 33, 35, 37, 39, 40, 42, 44,
          45, 47, 48, 50, 51, 53, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77,
          79, 80, 81, 82, 83, 85, 86, 87, 90, 91, 92, 94, 95, 96, 98, 100, 103, 104, 105, 108, 109, 118, 119, 120, 121,
          122, 123, 124, 127, 132, 135, 136, 137, 138, 141, 142, 144, 146, 148, 149, 150, 151, 152, 155, 156, 157, 158,
          159, 160, 161, 162, 163, 164, 165, 166, 169, 170, 171, 172, 173, 174, 177, 178, 180, 182, 183, 184, 185, 186,
          188, 189, 190, 191, 192, 194, 196, 200, 201, 202, 205, 206, 207, 208, 210, 211, 212, 213, 215, 216, 217, 218,
          219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
          243, 244, 246, 248, 249, 250, 251, 252, 253, 254, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 270,
          272, 273, 276, 277, 278, 280, 283, 284, 285, 287, 292, 293, 300, 204, 296, 297]


def resize_depth(data, std_depth=144):
    # data : npz file
    wide, height, depth = data.shape
    diff = std_depth - depth
    term = std_depth / diff
    string = np.arange(1, std_depth-1, term)
    string = string.astype(int)

    output = np.empty((wide, height, std_depth))
    output[:] = np.nan
    origin_idx = 0
    string_idx = 0
    idx = 0

    while idx < std_depth:
        try:
            if idx != string[string_idx]:
                output[:, :, idx] = data[:, :, origin_idx]
                idx += 1
                origin_idx += 1
            else:
                # output[:, :, idx] = (data[:, :, origin_idx - 1] + data[:, :, origin_idx]) / 2
                output[:, :, idx] = np.mean((data[:, :, origin_idx - 1], data[:, :, origin_idx]), axis=0)
                idx += 1
                string_idx += 1
        except IndexError:
            output[:, :, idx] = data[:, :, origin_idx]
            idx += 1
            origin_idx += 1

    return output


def align_mat(im, imRef):
    # im & imRef's is 2d-array
    imRefPart = imRef[-40:, :]
    im = im / 4096 * 255
    imRefPart = imRefPart / 4096 * 255
    im = im.astype(np.uint8)

    sums = []
    for i in range(-20, 21):
        translation_mat = np.float32([[1, 0, 0], [0, 1, i]])
        imMv = cv2.warpAffine(im, translation_mat, (im.shape[1], im.shape[0]))
        imPartmv = imMv[-40:, :]
        imPartmv = imPartmv.astype(np.float64)
        diff_mat = abs(imRefPart - imPartmv)
        sums.append(diff_mat.sum())
    mv_idx = sums.index(min(sums)) - 20

    sums = []
    for i in range(-10, 11):
        translation_mat = np.float32([[1, 0, i], [0, 1, mv_idx]])
        imMv = cv2.warpAffine(im, translation_mat, (im.shape[1], im.shape[0]))
        imPartmv = imMv[-40:, :]
        imPartmv = imPartmv.astype(np.float64)
        diff_mat = abs(imRefPart - imPartmv)
        sums.append(diff_mat.sum())
    mv_idy = sums.index(min(sums)) - 10

    if mv_idx >= 0:
        h = np.float32([[1, 0, mv_idy], [0, 1, mv_idx]])
    else:
        h = np.float32([[1, 0, -mv_idy], [0, 1, -mv_idx]])

    return h, mv_idx, mv_idy


def eval_gen(data_dir, vol_names, alignment=False):
    for vol_name in tqdm(vol_names):
        volumes = io.loadmat(os.path.join(data_dir, vol_name))
        pre = volumes['imgV_pre']
        delay = volumes['imgV_delay']
        pre = measure.block_reduce(pre, (2, 2, 1), np.max)
        delay = measure.block_reduce(delay, (2, 2, 1), np.max)

        if alignment:
            h, idx, idy = align_mat(pre[:, :, 0], delay[:, :, 0])

            if idx > 0:
                for i in range(pre.shape[2]):
                    pre[:, :, i] = cv2.warpAffine(pre[:, :, i], h, (256, 256))

            elif idx < 0:
                for i in range(delay.shape[2]):
                    delay[:, :, i] = cv2.warpAffine(delay[:, :, i], h, (256, 256))

        pre = resize_depth(pre)[np.newaxis, ..., np.newaxis]
        delay = resize_depth(delay)[np.newaxis, ..., np.newaxis]
        pre = pre / 4096
        delay = delay / 4096

        yield pre, delay, vol_name.split('.')[0]


def test(expr_name, epoch, gpu_id,
         nf_enc=(16, 32, 32, 32), nf_dec=(32, 32, 32, 32, 32, 16, 16)):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """
    vol_size = (256, 256, 144)

    # gpu handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)
        print("model load weights")
        net.load_weights(os.path.join("../models", expr_name, "{:04d}.h5".format(epoch)))

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # load subject test
    data_dir = '../../../dataset/urinary'

    except_list = [6, 97, 125, 198, 199, 228, 271]
    # vol_names = [filename for filename in os.listdir(data_dir) if (int(filename.split("_")[-1].split('.')[0]) <= 10) and
    #                                                           (int(filename.split("_")[-1].split('.')[0]) not in except_list)]

    vol_names = [filename for filename in os.listdir(data_dir) if int(filename.split("_")[-1].split(".")[0]) in normal]
    vol_names.sort()
    print("data length:", len(vol_names))
    vol_names = vol_names[:10]

    generator = eval_gen(data_dir=data_dir, vol_names=vol_names)

#    # if CPU, prepare grid
#    if compute_type == 'CPU':
#        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)

    if not os.path.isdir(os.path.join('../result', '{}_epoch{}'.format(expr_name, epoch))):
        os.mkdir(os.path.join('../result', '{}_epoch{}'.format(expr_name, epoch)))

    for pre, delay, vol_name in generator:
        with tf.device(gpu):
            pred = net.predict([pre, delay])
            X_warp = nn_trf_model.predict([pre, pred[1]])[0, ..., 0]

            pre = np.flip(pre, 3)[0, ..., 0]
            delay = np.flip(delay, 3)[0, ..., 0]
            X_warp = np.flip(X_warp, 2)

            pre = np.rot90(pre, 3)
            delay = np.rot90(delay, 3)
            X_warp = np.rot90(X_warp, 3)

            pre = pre * 4096 - 1024
            pre = pre.astype(np.int16)
            delay = delay * 4096 - 1024
            delay = delay.astype(np.int16)
            X_warp = X_warp * 4096 - 1024
            X_warp = X_warp.astype(np.int16)

            warp_img = nib.Nifti1Image(X_warp, affine=np.eye(4))
            nib.save(warp_img, os.path.join('../result', '{}_epoch{}'.format(expr_name, epoch),
                                            "{}_registration.nii.gz".format(vol_name)))

            moving_img = nib.Nifti1Image(pre, affine=np.eye(4))
            nib.save(moving_img, os.path.join('../result', '{}_epoch{}'.format(expr_name, epoch),
                                              "{}_pre.nii.gz".format(vol_name)))

            fixed_img = nib.Nifti1Image(delay, affine=np.eye(4))
            nib.save(fixed_img, os.path.join('../result', '{}_epoch{}'.format(expr_name, epoch),
                                             "{}_delay.nii.gz".format(vol_name)))

    print("successfully done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--expr_name', type=str,
                        dest='expr_name')
    parser.add_argument('--epoch', type=int,
                        dest='epoch')
    parser.add_argument('--gpu', type=int,
                        dest='gpu_id')

    test(**vars(parser.parse_args()))
