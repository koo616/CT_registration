# py imports
import os
import sys

# third party
import tensorflow as tf
import numpy as np
import nibabel as nib
from keras.backend.tensorflow_backend import set_session
from skimage import measure

# project
sys.path.append('../ext/medipy-lib')
import networks
from medipy.metrics import dice


def resize_depth(data, std_depth=112):
    # data : npz file
    wide, height, depth = data.shape
    diff = std_depth - depth
    term = std_depth / diff
    string = np.arange(1,std_depth-1,term)
    string = string.astype(int)

    for i in string:
        tmp = np.empty((wide, height))
        tmp[:] = np.nan
        data = np.insert(data, i, tmp, axis=2)

    for i in np.arange(data.shape[2]):
        if np.isnan(data[0, 0, i]):
            data[:, :, i] = (data[:, :, i-1] + data[:, :, i+1]) / 2

    return data


def load_data(data_dir, mode='test', fixed='joyoungje'):

    path = os.path.join(data_dir, mode)
    volumes = [np.load(os.path.join(path, filename)) for filename in os.listdir(path) if '.npz' in filename]

    moving_img = []
    fixed_img = []

    volume = volumes[0]
    joyoungje = volume['joyoungje']
    nojoyoungje = volume['nojoyoungje']
    if fixed == 'joyoungje':
        moving_img.append(resize_depth(nojoyoungje)[np.newaxis, ..., np.newaxis])
        fixed_img.append(resize_depth(joyoungje)[np.newaxis, ..., np.newaxis])
    else:
        moving_img.append(resize_depth(joyoungje)[np.newaxis, ..., np.newaxis])
        fixed_img.append(resize_depth(nojoyoungje)[np.newaxis, ..., np.newaxis])
    moving_img = np.concatenate(moving_img, axis=0)  # (number of data, w, h, depth, channel)
    moving_img = measure.block_reduce(moving_img, (1, 2, 2, 1, 1), np.max)
    fixed_img = np.concatenate(fixed_img, axis=0)
    fixed_img = measure.block_reduce(fixed_img, (1, 2, 2, 1, 1), np.max)
    moving_img = moving_img.astype('int32')
    fixed_img = fixed_img.astype('int32')

    return moving_img, fixed_img


def test(model_name, gpu_id,
         nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,32,16,16]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

    # load subject test
    print("load_data start")
    X_train, y_train = load_data(data_dir='../data', mode='test', fixed='joyoungje')
    vol_size = y_train.shape[1:-1]

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
        net.load_weights(model_name)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

#    # if CPU, prepare grid
#    if compute_type == 'CPU':
#        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)

    with tf.device(gpu):
        print("model predict")
        pred = net.predict([X_train, y_train])
        print("nn_tft_model.predict")
        X_warp = nn_trf_model.predict([X_train, pred[1]])[0,...,0]

    reshape_y_train = y_train.reshape(y_train.shape[1:-1])
    vals = dice(pred[0].reshape(pred[0].shape[1:-1]), reshape_y_train)
    dice_mean = np.mean(vals)
    dice_std = np.std(vals)
    print('Dice mean over structures: {:.2f} ({:.2f})'.format(dice_mean, dice_std))



if __name__ == "__main__":
    # test(sys.argv[1], sys.argv[2], sys.argv[3])
    test(sys.argv[1], sys.argv[2])
