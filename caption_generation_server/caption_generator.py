import numpy as np
from feature_extractor import get_features
from caption_generation_model.test import get_predict_function


predict_probs, word_to_idx, idx_to_word = get_predict_function('data/models/first_attempt/model_epoch_2_iter_240000.pckl')


def generate_captions(filename, top_n=5):
    cnn_features = get_features(filename)
    stop_idx = word_to_idx['<START/STOP>']

    beam_size = 20
    pool = [[[stop_idx], 0]]
    first_pass = True
    # n.b. this is suboptimal a lot of recalculations occur
    while any(each[0][-1] != stop_idx for each in pool) or first_pass:
        new_pool = []
        for entry in pool:
            if entry[0][-1] == stop_idx and not first_pass:
                new_pool.append(entry)
                continue
            probs = predict_probs(entry[0], cnn_features).flatten()
            for word_idx in probs.argsort()[-beam_size:]:
                k = len(entry[0]) - 1
                caption = entry[0] + [word_idx]
                score = (entry[1] * k - np.log(probs[word_idx])) / (k + 1)
                new_pool.append([caption, score])
        pool = sorted(new_pool, key=lambda e: e[1])[:beam_size]
        first_pass = False
        if beam_size != 5:
            beam_size -= 1
    pool = sorted(pool, key=lambda e: e[1])[:top_n]

    for entry in pool:
        entry[1] *= len(entry[0]) - 1
        entry[0] = u' '.join([idx_to_word[word_idx] for word_idx in entry[0][1:-1]])
    pool = sorted(pool, key=lambda e: e[1])
    return zip(*pool)