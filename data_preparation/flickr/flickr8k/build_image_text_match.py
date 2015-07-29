import json
import os.path
from collections import defaultdict


flickr8k_captions_files = "data/datasets/Flickr8k/Flickr8k_text/Flickr8k.token.txt"
flickr8k_image_folder = "data/datasets/Flickr8k/Flickr8k_Dataset"
flickr8k_image_captions_file = 'data/datasets/Flickr8k/flickr8k.json'


image_path_to_captions = defaultdict(list)
with open(flickr8k_captions_files) as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        image_name, caption = line.split('\t')
        image_name, caption_num = image_name.split('#')

        image_path = flickr8k_image_folder + '/' + image_name
        if not os.path.isfile(image_path):
            print 'There is no such file {}'.format(image_path)
            continue
        image_path_to_captions[image_path].append(caption.split())


with open(flickr8k_image_captions_file, 'w') as f:
    json.dump(image_path_to_captions.items(), f)