import numpy as np

from neural.utils import zero_pad
from .layer import Layer

class Conv(Layer):
    trainable = True

    def __init__(self, filter_count, filter_shape, stride, pad="valid"):
        (f_r, f_c) = filter_shape

        if pad == "same":
            self.pad = (f_r-1)//2
        elif pad == "valid":
            self.pad = 0
        elif not isinstance(pad, int):
            raise ValueError("Unrecognized input for pad.")
        else:
            self.pad = pad

        self.filter_count = filter_count
        self.filter_shape = filter_shape
        self.stride = stride
    
    def _initialize(self, input_dim):
        (X_r, X_c, X_ch) = input_dim
        (f_r, f_c) = self.filter_shape
        out_h = int((X_r + 2*self.pad - f_r)/self.stride + 1)
        out_w = int((X_c + 2*self.pad - f_c)/self.stride + 1)
        self.output_dim = (out_h, out_w, self.filter_count)

        self.W = np.random.randn(*self.filter_shape, X_ch, self.filter_count) * 0.1
        self.b = np.zeros((1, 1, 1, self.filter_count))

    def _forward(self, X, training=True):
        (m, X_r, X_c, X_ch) = X.shape
        (f_r, f_c, f_ch, n_f) = self.W.shape

        if X_ch != f_ch:
            raise ValueError("Input channels of the conv layer and number of channels of the input must match.")
        
        if training:
            self.X = X

        (out_h, out_w, n_f) = self.output_dim
        O = np.zeros((m, out_h, out_w, n_f))

        X_pad = zero_pad(X, self.pad)
        W_unsqueezed = self.W[np.newaxis]

        for r in range(out_h):
            for c in range(out_w):
                r_start = r*self.stride
                r_end = r_start + f_r
                c_start = c*self.stride
                c_end = c_start + f_c

                X_pad_slice = X_pad[:, r_start:r_end, c_start:c_end, :, np.newaxis]
                O[:, r, c, :] = np.sum(X_pad_slice * W_unsqueezed, axis=(1, 2, 3))
        
        O += self.b
        return O

    def _backward(self, dO):
        (m, X_r, X_c, X_ch) = self.X.shape
        (f_r, f_c, f_ch, n_f) = self.W.shape
        (m, out_h, out_w, n_f) = dO.shape

        dX = np.zeros(self.X.shape)
        self.dW = np.zeros(self.W.shape)
        self.db = np.sum(dO, axis=(0, 1, 2)).reshape(*self.b.shape)

        X_pad = zero_pad(self.X, self.pad)
        dX_pad = zero_pad(dX, self.pad)
        W_unsqueezed = self.W[np.newaxis]

        for r in range(out_h):
            for c in range(out_w):
                r_start = r*self.stride
                r_end = r_start + f_r
                c_start = c*self.stride
                c_end = c_start + f_c

                X_pad_curr = X_pad[:, r_start:r_end, c_start:c_end, :, np.newaxis]
                dO_curr = np.expand_dims(dO[:, r, c, np.newaxis, :], axis=(1, 2))
                dX_pad[:, r_start:r_end, c_start:c_end, :] += np.sum(dO_curr * W_unsqueezed, axis=4)
                self.dW += np.sum(X_pad_curr * dO_curr, axis=0)

        if self.pad > 0:
            dX = dX_pad[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dX = dX_pad

        return dX