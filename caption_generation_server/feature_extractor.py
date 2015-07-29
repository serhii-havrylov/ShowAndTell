import caffe
import numpy as np
from threading import Lock
from scipy.misc import imread, imresize


model_def_file = 'data/VGG/VGG_ILSVRC_16_layers_features_deploy.prototxt'
pretrained_model_file = 'data/VGG/VGG_ILSVRC_16_layers.caffemodel'
mean = np.array([103.939, 116.779, 123.68])
image_dim = 224
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
# caffe.set_device(0)
net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
net_lock = Lock()
batch_size, channels_num, height, weight = net.blobs[net.inputs[0]].data.shape
features_dim = net.blobs[net.outputs[0]].data.shape[1]


def get_features(fname):
    net_lock.acquire()
    try:
        im = imread(fname)
        if len(im.shape) == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
        # resize
        im = imresize(im, (height, weight))
        # RGB -> BGR
        im = im[..., (2, 1, 0)]
        # mean subtraction
        im = im - mean
        # get channel in correct dimension
        im = np.transpose(im, (2, 0, 1))

        in_data = np.zeros((batch_size, channels_num, height, weight), dtype=np.float32)
        in_data[0, ...] = im

        out = net.forward(**{net.inputs[0]: in_data})
        return out[net.outputs[0]][0]
    finally:
        net_lock.release()