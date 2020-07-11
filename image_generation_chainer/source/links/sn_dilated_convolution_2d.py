import chainer
import numpy as np
from chainer import cuda
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import convolution_2d
from chainer.links.connection.convolution_2d import Convolution2D
from source.functions.max_sv import max_singular_value

from chainer.functions.connection import dilated_convolution_2d
from chainer import initializers
from chainer import link
from chainer import variable
from chainer.links.connection.dilated_convolution_2d import DilatedConvolution2D


class SNDilatedConvolution2D(DilatedConvolution2D):
    """Two-dimensional dilated convolutional layer with spectral normalization.
    Please see the original DilatedConv implementation:
    https://github.com/chainer/chainer/blob/v3.2.0/chainer/links/connection/dilated_convolution_2d.py
    """

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 dilate=1, nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1,
                 adjust_snorm_for_convfilter=False):

        if adjust_snorm_for_convfilter is not False:
            self.ratio = ksize // stride
        else:
            self.ratio = None

        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.out_channels = out_channels
        self.use_gamma = use_gamma
        self.Ip = Ip
        self.u = None

        super(SNDilatedConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad, dilate,
            nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        xp = cuda.get_array_module(self.W.data)
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = self.ratio * sigma if self.ratio is not None else sigma
        sigma = broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        self.u = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_channels):
        super(SNDilatedConvolution2D, self)._initialize_params(in_channels)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        """Applies the convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of the convolution.
        """
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return dilated_convolution_2d.dilated_convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad, self.dilate)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
