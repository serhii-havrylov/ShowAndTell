import json
import numpy as np


files = {'mscoco': 'data/datasets/mscoco/mscoco.json',
         'flickr30k': 'data/datasets/Flickr30k/flickr30k.json',
         'flickr8k': 'data/datasets/Flickr8k/flickr8k.json'}
npa_paths = {'mscoco': {'train': 'data/datasets/mscoco/mscoco_train.npy',
                        'val': 'data/datasets/mscoco/mscoco_val.npy'},
             'flickr30k': 'data/datasets/Flickr30k/flickr30k.npy',
             'flickr8k': 'data/datasets/Flickr8k/flickr8k.npy'}


merged_image_captions_file_path = {'val': 'data/datasets/merged_val.json',
                                   'train': 'data/datasets/merged_train.json'}
merged_features_file_path = {'val': 'data/datasets/merged_val.npy',
                             'train': 'data/datasets/merged_train.npy'}


if __name__:
    with open(files['mscoco']) as f:
        image_captions = json.load(f)
        image_captions_mscoco_train = image_captions['train']
        image_captions_mscoco_valid = image_captions['val']
        features_mscoco_train = np.load(npa_paths['mscoco']['train'])
        features_mscoco_val = np.load(npa_paths['mscoco']['val'])

    with open(files['flickr30k']) as f:
        image_captions_flickr30k = json.load(f)
        features_flickr30k = np.load(npa_paths['flickr30k'])

    with open(files['flickr8k']) as f:
        image_captions_flickr8k = json.load(f)
        features_flickr8k = np.load(npa_paths['flickr8k'])

    image_captions_merged_train = []
    merged_features_train = []
    image_captions_merged_val = []
    merged_features_val = []

    image_captions_merged_train += image_captions_mscoco_train
    merged_features_train.append(features_mscoco_train)
    image_captions_merged_train += image_captions_mscoco_valid[:-17000]
    merged_features_train.append(features_mscoco_val[:-17000])
    image_captions_merged_train += image_captions_flickr30k[:-2000]
    merged_features_train.append(features_flickr30k[:-2000])
    image_captions_merged_train += image_captions_flickr8k[:-1000]
    merged_features_train.append(features_flickr8k[:-1000])

    image_captions_merged_val += image_captions_mscoco_valid[-17000:]
    merged_features_val.append(features_mscoco_val[-17000:])
    image_captions_merged_val += image_captions_flickr30k[-2000:]
    merged_features_val.append(features_flickr30k[-2000:])
    image_captions_merged_val += image_captions_flickr8k[-1000:]
    merged_features_val.append(features_flickr8k[-1000:])

    with open(merged_image_captions_file_path['train'], 'w') as f:
        json.dump(image_captions_merged_train, f)
    with open(merged_image_captions_file_path['val'], 'w') as f:
        json.dump(image_captions_merged_val, f)
    np.save(merged_features_file_path['train'], np.vstack(merged_features_train))
    np.save(merged_features_file_path['val'], np.vstack(merged_features_val))