import theano
import numpy as np
import theano.tensor as T


class LstmLayer(object):
    def __init__(self, name, W_z_init, W_i_init, W_f_init, W_o_init,
                             R_z_init, R_i_init, R_f_init, R_o_init,
                             b_z_init, b_i_init, b_f_init, b_o_init):
        self.name = name

        W_init = np.hstack((W_z_init(), W_i_init(), W_f_init(), W_o_init()))
        R_init = np.hstack((R_z_init(), R_i_init(), R_f_init(), R_o_init()))
        b_init = np.hstack((b_z_init(), b_i_init(), b_f_init(), b_o_init()))
        self.W = theano.shared(W_init, name=name + '_W_zifo')
        self.R = theano.shared(R_init, name=name + '_R_zifo')
        self.b = theano.shared(b_init, name=name + '_b_zifo')
        self.n = b_init.shape[0] / 4

    def get_output_expr(self, input_sequence):
        h0 = T.zeros((self.n, ), dtype=np.float32)
        c0 = T.zeros((self.n, ), dtype=np.float32)

        [_, h], _ = theano.scan(fn=self.__get_lstm_step_expr,
                                sequences=input_sequence,
                                outputs_info=[c0, h0])
        return h

    def __get_lstm_step_expr(self, x_t, c_tm1, h_tm1):
        sigm = T.nnet.sigmoid
        tanh = T.tanh
        dot = theano.dot

        zifo_t = dot(x_t, self.W) + dot(h_tm1, self.R) + self.b
        z_t = tanh(zifo_t[0*self.n:1*self.n])
        i_t = sigm(zifo_t[1*self.n:2*self.n])
        f_t = sigm(zifo_t[2*self.n:3*self.n])
        o_t = sigm(zifo_t[3*self.n:4*self.n])

        c_t = i_t * z_t + f_t * c_tm1
        h_t = o_t * tanh(c_t)
        return c_t, h_t

    def get_parameters(self):
        return [self.W, self.R, self.b]

    @classmethod
    def create_copy(cls, other):
        self = cls.__new__(cls)
        self.name = other.name
        self.W = other.W
        self.R = other.R
        self.b = other.b
        self.n = other.n
        return self

    @classmethod
    def create_from_state(cls, state):
        self = cls.__new__(cls)
        self.name = state['name']
        self.W = theano.shared(state['W'])
        self.R = theano.shared(state['R'])
        self.b = theano.shared(state['b'])
        self.n = state['n']
        return self

    def get_state(self):
        state = {}
        state['name'] = self.name
        state['W'] = self.W.get_value()
        state['R'] = self.R.get_value()
        state['b'] = self.b.get_value()
        state['n'] = self.n
        return state