import time
import theano
import cPickle
import logging
import datetime
import numpy as np
from theano import tensor as T
from data_iterators import TrainDataIterator
from data_iterators import ValidationDataIterator
from caption_generation_model.model_blocks import nonlinearities
from caption_generation_model.model_blocks.layers import LstmLayer
from caption_generation_model.model_blocks.layers import DenseLayer
from caption_generation_model.model_blocks.layers import ScaleLayer
from caption_generation_model.model_blocks.layers import DropoutLayer
from caption_generation_model.model_blocks.layers import RowStackLayer
from caption_generation_model.model_blocks.layers import EmbeddingLayer
from caption_generation_model.model_blocks.layers import NonlinearityLayer
from caption_generation_model.model_blocks.initializers import Constant
from caption_generation_model.model_blocks.initializers import Normal
from caption_generation_model.model_blocks.initializers import Orthogonal
from caption_generation_model.model_blocks.learning_rules import get_nesterov_momentum_updates


def get_logger():
    logger = logging.getLogger('train_logger')
    handler = logging.FileHandler('data/models/second_attempt/train.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%d-%m-%Y %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
logger = get_logger()


# load train and val data
train_data_iterator = TrainDataIterator()
validation_data_iterator = ValidationDataIterator(train_data_iterator.idx_to_word,
                                                  train_data_iterator.word_to_idx)
logger.info('Vocab size: {}'.format(len(train_data_iterator.idx_to_word)))
logger.info('Train data size: {}'.format(len(train_data_iterator.data_ids)))
logger.info('Val data size: {}'.format(len(validation_data_iterator.data_ids)))


# define train model
vocab_size = len(train_data_iterator.idx_to_word)
cnn_features_dim = 4096
word_embed_dim = 1024
hidden_state_dim = 1024
normal_init = Normal((word_embed_dim, hidden_state_dim))
orthog_init = Orthogonal((hidden_state_dim, hidden_state_dim))
b_init = Constant((hidden_state_dim, ))


word_embedding_layer = EmbeddingLayer(name='word_embedding',
                                      embedding_init=Normal((vocab_size, word_embed_dim)))
cnn_embedding_layer = DenseLayer(name='cnn_embedding',
                                 W_init=Normal((cnn_features_dim, word_embed_dim)))
row_stack_layer = RowStackLayer('row_stack')
embedding_dropout_layer = DropoutLayer(name='embedding_dropout',
                                       dropout_prob=0.5)
lstm_layer = LstmLayer(name='lstm',
                       W_z_init=normal_init, W_i_init=normal_init, W_f_init=normal_init, W_o_init=normal_init,
                       R_z_init=orthog_init, R_i_init=orthog_init, R_f_init=orthog_init, R_o_init=orthog_init,
                       b_z_init=b_init,      b_i_init=b_init,      b_f_init=b_init,      b_o_init=b_init)
hidden_states_dropout_layer = DropoutLayer(name='hidden_states_dropout',
                                           dropout_prob=0.5)
pre_softmax_layer = DenseLayer(W_init=Constant((hidden_state_dim, vocab_size)),
                               b_init=Constant((vocab_size, )),
                               name='pre_softmax')
softmax_layer = NonlinearityLayer(name='softmax',
                                  nonlinearity=nonlinearities.softmax)


# define forward propagation expression for train model and loss function
learning_rate = theano.shared(np.float32(0.01))
word_indices = T.ivector()
cnn_features = T.fvector()
true_dist = T.ivector()

word_embedings = word_embedding_layer.get_output_expr(word_indices)
cnn_embedings = cnn_embedding_layer.get_output_expr(cnn_features)
embedings = row_stack_layer.get_output_expr(cnn_embedings, word_embedings)
masked_embedings = embedding_dropout_layer.get_output_expr(embedings)
h = lstm_layer.get_output_expr(masked_embedings)
masked_h = hidden_states_dropout_layer.get_output_expr(h[1:])
unnormalized_probs = pre_softmax_layer.get_output_expr(masked_h)
probs = softmax_layer.get_output_expr(unnormalized_probs)
loss = T.mean(T.nnet.categorical_crossentropy(probs, true_dist))
updates = get_nesterov_momentum_updates(loss_expr=loss,
                                        dense_parameters=cnn_embedding_layer.get_parameters() + \
                                                         row_stack_layer.get_parameters() + \
                                                         embedding_dropout_layer.get_parameters() + \
                                                         lstm_layer.get_parameters() + \
                                                         hidden_states_dropout_layer.get_parameters() + \
                                                         pre_softmax_layer.get_parameters() + \
                                                         softmax_layer.get_parameters(),
                                        sparse_parameters=word_embedding_layer.get_parameters(),
                                        learning_rate=learning_rate, momentum=0.9)


# compile model training function
cnn_features_idx = T.iscalar()
caption_begin = T.iscalar()
caption_end = T.iscalar()
train_model = theano.function(
        inputs=[cnn_features_idx, caption_begin, caption_end],
        outputs=loss,
        updates=updates,
        givens={
            cnn_features: train_data_iterator.features[cnn_features_idx],
            word_indices: train_data_iterator.flatten_input_captions[caption_begin:caption_end],
            true_dist: train_data_iterator.flatten_output_captions[caption_begin:caption_end]
        }
    )


# define valid model
val_word_embedding_layer = EmbeddingLayer.create_copy(word_embedding_layer)
val_cnn_embedding_layer = DenseLayer.create_copy(cnn_embedding_layer)
val_row_stack_layer = RowStackLayer.create_copy(row_stack_layer)
val_embedding_scale_layer = ScaleLayer('embedding_scale', embedding_dropout_layer.keep_prob)
val_lstm_layer = LstmLayer.create_copy(lstm_layer)
val_hidden_states_scale_layer = ScaleLayer('hidden_states_scale', hidden_states_dropout_layer.keep_prob)
val_pre_softmax_layer = DenseLayer.create_copy(pre_softmax_layer)
val_softmax_layer = NonlinearityLayer.create_copy(softmax_layer)


# define forward propagation expression for val model and loss function
val_word_embedings = val_word_embedding_layer.get_output_expr(word_indices)
val_cnn_embedings = val_cnn_embedding_layer.get_output_expr(cnn_features)
val_embedings = val_row_stack_layer.get_output_expr(val_cnn_embedings, val_word_embedings)
val_scaled_embedings = val_embedding_scale_layer.get_output_expr(val_embedings)
val_h = val_lstm_layer.get_output_expr(val_scaled_embedings)
val_scaled_h = val_hidden_states_scale_layer.get_output_expr(val_h[1:])
val_unnormalized_probs = val_pre_softmax_layer.get_output_expr(val_scaled_h)
val_probs = softmax_layer.get_output_expr(val_unnormalized_probs)
val_loss = T.mean(T.nnet.categorical_crossentropy(val_probs, true_dist))


def save_val_model(file_path):
    model_state = {}
    model_state['word_embedding_layer'] = val_word_embedding_layer.get_state()
    model_state['cnn_embedding_layer'] = val_cnn_embedding_layer.get_state()
    model_state['row_stack_layer'] = val_row_stack_layer.get_state()
    model_state['embedding_scale_layer'] = val_embedding_scale_layer.get_state()
    model_state['lstm_layer'] = val_lstm_layer.get_state()
    model_state['hidden_states_scale_layer'] = val_hidden_states_scale_layer.get_state()
    model_state['pre_softmax_layer'] = val_pre_softmax_layer.get_state()
    model_state['softmax_layer'] = softmax_layer.get_state()
    model_state['word_to_idx'] = train_data_iterator.word_to_idx
    model_state['idx_to_word'] = train_data_iterator.idx_to_word

    with open(file_path, 'wb') as f:
        cPickle.dump(model_state, f)


# compile model validation function
val_model = theano.function(
        inputs=[cnn_features_idx, caption_begin, caption_end],
        outputs=val_loss,
        givens={
            cnn_features: validation_data_iterator.features[cnn_features_idx],
            word_indices: validation_data_iterator.flatten_input_captions[caption_begin:caption_end],
            true_dist: validation_data_iterator.flatten_output_captions[caption_begin:caption_end]
        }
    )


for epoch in xrange(10):
    logger.info('=======epoch: {}======='.format(epoch))

    t = time.time()
    ls = []
    for i, (cnnf_idx, (c_begin, c_end)) in enumerate(train_data_iterator.get_epoch_iterator()):
        ls.append(train_model(cnnf_idx, c_begin, c_end))
        if i % 60000 == 0 and i != 0:
            message = 'epoch: {} iter: {} train_loss: {}'.format(epoch, i, np.mean(ls))
            logger.info(message)
            ls = []
            now = datetime.datetime.now().time()
            save_val_model('data/models/second_attempt/model_epoch_{}_iter_{}.pckl'.format(epoch, i))
    learning_rate.set_value(np.float32(learning_rate.get_value()/1.4))
    logger.info('train epoch took {} sec'.format(time.time() - t))
    save_val_model('data/models/second_attempt/model_epoch_{}.pckl'.format(epoch))

    t = time.time()
    ls = []
    for cnnf_idx, (c_begin, c_end) in validation_data_iterator.get_epoch_iterator():
        ls.append(val_model(cnnf_idx, c_begin, c_end))
    logger.info('epoch: {} val_loss: {}'.format(epoch, np.mean(ls)))
    logger.info('valid took {} sec'.format(time.time() - t))