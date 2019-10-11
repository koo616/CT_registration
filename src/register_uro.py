# py imports
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import nibabel as nib
from keras.backend.tensorflow_backend import set_session
from skimage import measure

# project
sys.path.append('../ext/medipy-lib')
import networks


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


def load_data(data_dir, sample_num, mode='test', fixed='joyoungje'):

    path = os.path.join(data_dir, mode)
    volumes = [np.load(os.path.join(path, filename)) for filename in os.listdir(path) if '.npz' in filename]

    moving_img = []
    fixed_img = []

    assert sample_num in [1, 2, 3], "saple_num must be 1 or 2 or 3, found {}".format(sample_num)

    volume = volumes[sample_num-1]
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

    # scaling
    moving_img = (moving_img + 1024) / 4096
    fixed_img = (fixed_img + 1024) / 4096

    return moving_img, fixed_img


def test(expr_name, epoch, gpu_id, sample_num, dataset,
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
    X_train, y_train = load_data(data_dir='../data', sample_num=sample_num, mode=dataset, fixed='joyoungje')
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
        net.load_weights(os.path.join("../models", expr_name, "{}.h5".format(epoch)))
        #net.load_weights(os.path.join("../models", "{}.h5".format(expr_name)))

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

#    # if CPU, prepare grid
#    if compute_type == 'CPU':
#        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)

    with tf.device(gpu):
        print("model predict")
        pred = net.predict([X_train, y_train])
        #print("nn_tft_model.predict")
        X_warp = nn_trf_model.predict([X_train, pred[1]])[0,...,0]

    print("saving...")
    if not os.path.isdir('../result/{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num)):
        os.mkdir('../result/{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num))

    warp_img = nib.Nifti1Image(X_warp, affine=np.eye(4))
    nib.save(warp_img, os.path.join('../result', '{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num), "registration.nii.gz"))

    # warp_img = nib.Nifti1Image(pred[0].reshape(pred[0].shape[1:-1]), affine=np.eye(4))
    # nib.save(warp_img, os.path.join('../result', '{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num), "registration.nii.gz"))

    moving_img = nib.Nifti1Image(X_train.reshape(X_train.shape[1:-1]), affine=np.eye(4))
    nib.save(moving_img, os.path.join('../result', '{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num), "original.nii.gz"))

    fixed_img = nib.Nifti1Image(y_train.reshape(y_train.shape[1:-1]), affine=np.eye(4))
    nib.save(fixed_img, os.path.join('../result', '{}_epoch{}_{}{}'.format(expr_name, epoch, dataset, sample_num), "reference.nii.gz"))

    print("successfully done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--expr_name', type=str,
                        dest='expr_name')
    parser.add_argument('--epoch', type=int,
                        dest='epoch')
    parser.add_argument('--gpu_id', type=int,
                        dest='gpu_id')
    parser.add_argument('--dataset', type=str,
                        dest='dataset', default='test')
    parser.add_argument('--sample_num', type=int,
                        dest='sample_num', default=1)

    test(**vars(parser.parse_args()))
