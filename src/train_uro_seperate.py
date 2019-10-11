# python imports
import os
import cv2
from argparse import ArgumentParser
from tqdm import tqdm
import random

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from skimage import measure
from scipy import io
import time

# project imports
import networks
import losses

except_list = [6, 97, 125, 198, 199, 228, 271]

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


def uro_generator(vol_names, path, shuffle=True, fixed='joyoungje'):
    """ generator used for urography model """
    volumes = []
    length = len(vol_names)
    for vol_name in tqdm(vol_names):
        # print("io start")
        each_vol = io.loadmat(os.path.join(path, vol_name))
        # print("align start")

        yes_jo = measure.block_reduce(each_vol['imgV_delay'], (2, 2, 1), np.max)
        no_jo = measure.block_reduce(each_vol['imgV_pre'], (2, 2, 1), np.max)

        if False:
            h, idx, idy = align_mat(no_jo[:, :, 0], yes_jo[:, :, 0])
            if idx > 0:
                for i in range(no_jo.shape[2]):
                    no_jo[:, :, i] = cv2.warpAffine(no_jo[:, :, i], h, (256, 256))

                # cv2.imwrite("{}_ref.png".format(vol_name), (yes_jo[:, :, 0] / 4096 * 255).astype(np.uint8))
                # cv2.imwrite("{}_mv.png".format(vol_name), (no_jo[:, :, 0] / 4096 * 255).astype(np.uint8))
                #
                # print(idx, idy)

            elif idx < 0:
                for i in range(yes_jo.shape[2]):
                    yes_jo[:, :, i] = cv2.warpAffine(yes_jo[:, :, i], h, (256, 256))

                # cv2.imwrite("{}_ref.png".format(vol_name), (yes_jo[:, :, 0] / 4096 * 255).astype(np.uint8))
                # cv2.imwrite("{}_mv.png".format(vol_name), (no_jo[:, :, 0] / 4096 * 255).astype(np.uint8))

                # print(idx, idy)

        yes_jo = resize_depth(yes_jo)[np.newaxis, ..., np.newaxis]
        no_jo = resize_depth(no_jo)[np.newaxis, ..., np.newaxis]
        # print("block reduce")

        # scaling
        yes_jo = yes_jo / 4096
        no_jo = no_jo / 4096

        each_vol['imgV_delay'] = yes_jo
        each_vol['imgV_pre'] = no_jo
        volumes.append(each_vol)
    i = 0
    zeros = np.zeros(shape=yes_jo.shape)
    if shuffle:
        random.seed(42)
    while True:
        if shuffle and (i == 0):
            random.shuffle(volumes)
            print("shuffle!")
        volume = volumes[i]
        i = (i + 1) % length
        joyoungje = volume['imgV_delay']
        nojoyoungje = volume['imgV_pre']

        if fixed == 'joyoungje':
            yield ([nojoyoungje, joyoungje], [joyoungje, zeros])
        else:
            yield ([joyoungje, nojoyoungje], [nojoyoungje, zeros])


def train(data_dir,
          model,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          reg_param,
          steps_per_epoch,
          load_model_file,
          data_loss,
          window_size,
          batch_size):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc'
    """

    vol_size = [256, 256, 144]  # (width, height, depth)

    # set encoder, decoder feature number
    nf_enc = [16, 32, 32, 32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    else:  # 'vm2double':
        nf_enc = [f * 2 for f in nf_enc]
        nf_dec = [f * 2 for f in [32, 32, 32, 32, 32, 16, 16]]

    # set loss function
    # Mean Squared Error, Cross-Correlation, Negative Cross-Correlation
    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        NCC = losses.NCC(win=window_size)
        data_loss = NCC.loss

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    gpu = '/gpu:%d' % 0  # gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # prepare the model
    with tf.device(gpu):
        # in the CVPR layout, the model takes in [moving image, fixed image] and outputs [warped image, flow]
        model = networks.cvpr2018_net(vol_size, nf_enc, nf_dec)

        if load_model_file is not None:
            model.load_weights(load_model_file)

        # save first iteration
        # model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))

    # load data
    # path = "../../dataset/urinary"
    # vol_names = [filename for filename in os.listdir(data_dir) if (int(filename.split("_")[-1].split('.')[0]) < 206) and
    #                                                           (int(filename.split("_")[-1].split('.')[0]) not in except_list)]

    vol_names = [filename for filename in os.listdir(data_dir) if int(filename.split("_")[-1].split(".")[0]) in normal]

    # vol_names = [filename for filename in os.listdir(data_dir) if int(filename.split("_")[-1].split(".")[0]) in (9, 130, 128)]
    vol_names.sort()
    uro_gen = uro_generator(vol_names, data_dir, fixed='joyoungje')

    # test_path = os.path.join(data_dir, 'test')
    # test_vol_names = [filename for filename in os.listdir(test_path) if '.npz']
    # test_gen = uro_generator(test_vol_names, test_path)

    # fit
    with tf.device(gpu):
        mg_model = model

        # compile
        mg_model.compile(optimizer=Adam(lr=lr),
                         loss=[data_loss, losses.Grad('l2').loss],
                         loss_weights=[1.0, reg_param])

        # fit
        save_file_name = os.path.join(model_dir, '{epoch:04d}.h5')
        save_callback = ModelCheckpoint(save_file_name)
        mg_model.fit_generator(uro_gen,
                               epochs=nb_epochs,
                               verbose=1,
                               callbacks=[save_callback],
                               steps_per_epoch=steps_per_epoch)

            # score = mg_model.evaluate_generator(test_gen, verbose=1, steps=3)
            # print(score)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        help="data folder", default='../../../dataset/urinary/')
    parser.add_argument("--model", type=str, dest="model",
                        choices=['vm1', 'vm2', 'vm2double'], default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/lr_1e-5_ncc_lambda_1/',
                        help="models folder")
    parser.add_argument("--gpu", type=str, default='6',
                        dest="gpu_id", help="gpu id number (or numbers separated by comma)")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=1.0,  # recommend 1.0 for ncc, 0.01 for mse
                        help="regularization parameter")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=207,
                        help="frequency of model saves")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default=None,
                        help="optional h5 model file to initialize with")
    parser.add_argument("--data_loss", type=str,
                        dest="data_loss", default='ncc',
                        help="data_loss: mse of ncc")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=None,
                        help="batch size. default 'None'.")
    parser.add_argument("--ncc_window_size", type=int,
                        dest="window_size", default=9,
                        help="Window size to compute local mean in NCC loss")

    args = parser.parse_args()
    train(**vars(args))
