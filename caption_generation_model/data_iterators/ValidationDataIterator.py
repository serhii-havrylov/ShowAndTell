import json
import theano
import numpy as np
from numpy import random


captions_file = 'data/datasets/merged_val.json'
features_file = 'data/datasets/merged_val.npy'


class ValidationDataIterator(object):
    def __init__(self, idx_to_word, word_to_idx):
        self.r = random.RandomState(seed=42)
        self.idx_to_word = idx_to_word
        self.word_to_idx = word_to_idx
        self.image_path_to_captions = None
        self.preprocess_data(captions_file)

        self.data_ids = []
        self.caption_margins = []
        flatten_input_captions = []
        flatten_output_captions = []
        k = 0
        for i, (image_path, captions) in enumerate(self.image_path_to_captions):
            self.caption_margins.append([])
            for j, caption in enumerate(captions):
                self.data_ids.append((i, j))
                caption = [self.word_to_idx[word] for word in caption]
                flatten_input_captions += [self.word_to_idx['<START/STOP>']] + caption
                flatten_output_captions += caption + [self.word_to_idx['<START/STOP>']]
                self.caption_margins[i].append((k, k + 1 + len(caption)))
                k += len(caption) + 1

        self.flatten_input_captions = theano.shared(np.array(flatten_input_captions, dtype=np.int32))
        self.flatten_output_captions = theano.shared(np.array(flatten_output_captions, dtype=np.int32))
        self.features = theano.shared(np.load(features_file))

    def preprocess_data(self, captions_file):
        with open(captions_file) as f:
            self.image_path_to_captions = json.load(f)

        for image_path, captions in self.image_path_to_captions:
            for i, caption in enumerate(captions):
                new_caption = []
                for word in caption:
                    word = word.lower()
                    if word in self.word_to_idx:
                        new_caption.append(word)
                    else:
                        new_caption.append('<UNK>')
                captions[i] = new_caption

    def get_epoch_iterator(self):
        self.r.shuffle(self.data_ids)
        for i, j in self.data_ids:
            yield i, self.caption_margins[i][j]