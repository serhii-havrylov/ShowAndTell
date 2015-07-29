import theano
import cPickle
from theano import tensor as T
from caption_generation_model.model_blocks.layers import LstmLayer
from caption_generation_model.model_blocks.layers import DenseLayer
from caption_generation_model.model_blocks.layers import ScaleLayer
from caption_generation_model.model_blocks.layers import RowStackLayer
from caption_generation_model.model_blocks.layers import EmbeddingLayer
from caption_generation_model.model_blocks.layers import NonlinearityLayer


def get_predict_function(file_path):
    with open(file_path) as f:
        model_state = cPickle.load(f)

    # define model architecture
    word_embedding_layer = EmbeddingLayer.create_from_state(model_state['word_embedding_layer'])
    cnn_embedding_layer = DenseLayer.create_from_state(model_state['cnn_embedding_layer'])
    row_stack_layer = RowStackLayer.create_from_state(model_state['row_stack_layer'])
    embedding_scale_layer = ScaleLayer.create_from_state(model_state['embedding_scale_layer'])
    lstm_layer = LstmLayer.create_from_state(model_state['lstm_layer'])
    hidden_states_scale_layer = ScaleLayer.create_from_state(model_state['hidden_states_scale_layer'])
    pre_softmax_layer = DenseLayer.create_from_state(model_state['pre_softmax_layer'])
    softmax_layer = NonlinearityLayer.create_from_state(model_state['softmax_layer'])

    # define forward propagation expression for model
    word_indices = T.ivector()
    cnn_features = T.fvector()
    word_embedings = word_embedding_layer.get_output_expr(word_indices)
    cnn_embedings = cnn_embedding_layer.get_output_expr(cnn_features)
    embedings = row_stack_layer.get_output_expr(cnn_embedings, word_embedings)
    scaled_embedings = embedding_scale_layer.get_output_expr(embedings)
    h = lstm_layer.get_output_expr(scaled_embedings)
    scaled_h = hidden_states_scale_layer.get_output_expr(h[h.shape[0]-1])
    unnormalized_probs = pre_softmax_layer.get_output_expr(scaled_h)
    probs = softmax_layer.get_output_expr(unnormalized_probs)

    predict_probs = theano.function(inputs=[word_indices, cnn_features],
                                    outputs=probs)

    return predict_probs, model_state['word_to_idx'], model_state['idx_to_word']