import json
import time
import caffe
import numpy as np
from scipy.misc import imread, imresize


files = {'flickr30k': 'data/datasets/Flickr30k/flickr30k.json',
         'flickr8k': 'data/datasets/Flickr8k/flickr8k.json'}
model_def_file = 'data/VGG/VGG_ILSVRC_16_layers_features_deploy.prototxt'
pretrained_model_file = 'data/VGG/VGG_ILSVRC_16_layers.caffemodel'
mean = np.array([103.939, 116.779, 123.68])
image_dim = 224
out = {'flickr30k': 'data/datasets/Flickr30k/flickr30k.npy',
       'flickr8k': 'data/datasets/Flickr8k/flickr8k.npy'}


def batch_predict(filenames, net):
    batch_size, channels_num, height, weight = net.blobs[net.inputs[0]].data.shape
    features_dim = net.blobs[net.outputs[0]].data.shape[1]
    files_num = len(filenames)

    features = np.zeros((files_num, features_dim), dtype=np.float32)
    for i in xrange(0, files_num, batch_size):
        in_data = np.zeros((batch_size, channels_num, height, weight), dtype=np.float32)
        N = min(i + batch_size, files_num)
        Nb = N - i

        for j, f_path in enumerate(filenames[i:N]):
            im = imread(f_path)
            if len(im.shape) == 2:
                print f_path, 'has only two dimensions!'
                im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

            # resize
            im = imresize(im, (height, weight))
            # RGB -> BGR
            im = im[..., (2, 1, 0)]
            # mean subtraction
            im = im - mean
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            in_data[j, ...] = im

        out = net.forward(**{net.inputs[0]: in_data})
        features[i:N] = out[net.outputs[0]][:Nb]

        print 'Done %d/%d files' % (i+Nb, len(filenames))

    return features


if __name__ == '__main__':
    caffe.set_mode_gpu()
    net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)

    for flickr_type, image_captions_file_path in files.iteritems():
        print 10 * '=', flickr_type, 10 * '='
        t = time.time()
        print image_captions_file_path
        with open(image_captions_file_path) as f:
            image_captions = json.load(f)

        image_file_paths = zip(*image_captions)[0]
        features = batch_predict(image_file_paths, net)
        print time.time() - t
        print (time.time() - t) / len(image_file_paths)
        np.save(out[flickr_type], features)