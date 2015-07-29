import json
import os.path
from collections import defaultdict


flickr30k_captions_files = "data/datasets/Flickr30k/results_20130124.token"
flickr30k_image_folder = "data/datasets/Flickr30k/flickr30k-images"
flickr30k_image_captions_file = 'data/datasets/Flickr30k/flickr30k.json'


image_path_to_captions = defaultdict(list)
with open(flickr30k_captions_files) as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        image_name, caption = line.split('\t')
        image_name, caption_num = image_name.split('#')

        image_path = flickr30k_image_folder + '/' + image_name
        if not os.path.isfile(image_path):
            print 'There is no such file {}'.format(image_path)
            continue
        image_path_to_captions[image_path].append(caption.split())


with open(flickr30k_image_captions_file, 'w') as f:
    json.dump(image_path_to_captions.items(), f)