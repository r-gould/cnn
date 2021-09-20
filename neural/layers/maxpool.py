import numpy as np

from neural.utils import zero_pad
from .layer import Layer

class MaxPool(Layer):
    trainable = False

    def __init__(self, size, stride=1, pad="valid"):
        (size_r, size_c) = size

        if pad == "same":
            self.pad = (f_r-1)//2
        elif pad == "valid":
            self.pad = 0
        elif not isinstance(pad, int):
            raise ValueError("Unrecognized input for pad.")
        else:
            self.pad = pad

        self.size = size
        self.stride = stride

    def _initialize(self, input_dim):
        (X_r, X_c, X_ch) = input_dim
        (size_r, size_c) = self.size
        out_h = int((X_r + 2*self.pad - size_r)/self.stride + 1)
        out_w = int((X_c + 2*self.pad - size_c)/self.stride + 1)
        self.output_dim = (out_h, out_w, X_ch)

    def _forward(self, X, training=True):
        (size_r, size_c) = self.size
        (m, X_r, X_c, X_ch) = X.shape

        (out_h, out_w, X_ch) = self.output_dim
        O = np.zeros((m, out_h, out_w, X_ch))
        
        if training:
            self.X = X

        X_pad = zero_pad(X, self.pad)
        self.mask = np.zeros(X_pad.shape)

        for r in range(out_h):
            for c in range(out_w):
                r_start = r*self.stride
                r_end = r_start + size_r
                c_start = c*self.stride
                c_end = c_start + size_c

                X_pad_slice = X_pad[:, r_start:r_end, c_start:c_end, :]
                O[:, r, c, :] = np.max(X_pad_slice, axis=(1, 2))

                if training:
                    X_pad_slice_squeezed = np.reshape(X_pad_slice, (m, -1, X_ch))
                    max_idx = np.argmax(X_pad_slice_squeezed, axis=1)
                    (r_idx, c_idx) = np.indices((m, X_ch))

                    curr_mask_squeezed = np.zeros(X_pad_slice_squeezed.shape)
                    curr_mask_squeezed[r_idx, max_idx, c_idx] = 1
                    curr_mask = np.reshape(curr_mask_squeezed, X_pad_slice.shape)
                    self.mask[:, r_start:r_end, c_start:c_end, :] = curr_mask

        return O

    def _backward(self, dO):
        (size_r, size_c) = self.size
        (m, out_h, out_w, X_ch) = dO.shape
        dX = np.zeros(self.X.shape)
        dX_pad = zero_pad(dX, self.pad)

        for r in range(out_h):
            for c in range(out_w):
                r_start = r*self.stride
                r_end = r_start + size_r
                c_start = c*self.stride
                c_end = c_start + size_c

                curr_mask = self.mask[:, r_start:r_end, c_start:c_end, :]
                dO_curr = np.expand_dims(dO[:, r, c, :], axis=(1, 2))
                dX_pad[:, r_start:r_end, c_start:c_end, :] += dO_curr * curr_mask

        if self.pad > 0:
            dX = dX_pad[:, :, self.pad:-self.pad, self.pad:-self.pad]
        else:
            dX = dX_pad

        return dX