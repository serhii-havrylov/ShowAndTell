import json
import nltk
import os.path
from collections import defaultdict


mscoco_captions_files = {'train': 'data/datasets/mscoco/annotations/captions_train2014.json',
                         'val': 'data/datasets/mscoco/annotations/captions_val2014.json'}
mscoco_image_folder = {'train': 'data/datasets/mscoco/train2014',
                       'val': 'data/datasets/mscoco/val2014',
                       'test2014': 'data/datasets/mscoco/test2014'}
mscoco_image_captions_file = 'data/datasets/mscoco/mscoco.json'


image_path_to_captions = defaultdict(lambda: defaultdict(list))
for type in mscoco_captions_files:
    with open(mscoco_captions_files[type]) as f:
        mscoco = json.load(f)
        id_to_image_path = {}
        for image_info in mscoco['images']:
            image_path = mscoco_image_folder[type] + '/' + image_info['file_name']
            if not os.path.isfile(image_path):
                print 'There is no such file {}'.format(image_path)
                continue
            if image_info['id'] in id_to_image_path:
                print 'Id {} already exists!'.format(image_info['id'])
                continue
            id_to_image_path[image_info['id']] = image_path

        for annotaion_info in mscoco['annotations']:
            image_path = id_to_image_path[annotaion_info['image_id']]
            caption = nltk.word_tokenize(annotaion_info['caption'])
            image_path_to_captions[type][image_path].append(caption)

        image_path_to_captions[type] = image_path_to_captions[type].items()


with open(mscoco_image_captions_file, 'w') as f:
    json.dump(image_path_to_captions, f)