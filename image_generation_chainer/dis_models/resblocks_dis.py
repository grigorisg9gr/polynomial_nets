import math
import chainer
from source.links.sn_convolution_2d import SNConvolution2D
from chainer import functions as F
from chainer import links as L


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, dropout_ratio=0.0, sn=False,
                 gate=False):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.gate = gate
        if sn:
            Conv = SNConvolution2D
        else:
            Conv = L.Convolution2D
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = Conv(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = Conv(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = Conv(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
            if gate:
                self.a = chainer.Parameter(initializer=chainer.initializers.Zero(), shape=(1))

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        if self.gate:
            res = self.residual(x)
            sh = self.shortcut(x)
            sigm_a = F.broadcast_to(F.sigmoid(self.a)[:, None, None, None], res.shape)
            return 2 * ((1 - sigm_a) * res + sigm_a * sh)
        else:
            return self.residual(x) + self.shortcut(x)


class OptimizedBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1,
                 activation=F.relu, sn=False, gate=False):
        super(OptimizedBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.gate = gate
        if sn:
            Conv = SNConvolution2D
        else:
            Conv = L.Convolution2D
        with self.init_scope():
            self.c1 = Conv(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = Conv(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = Conv(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
            if gate:
                self.a = chainer.Parameter(initializer=chainer.initializers.Zero(), shape=(1))

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        # Conv -> Pool
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        # Pool -> Conv
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        if self.gate:
            res = self.residual(x)
            sh = self.shortcut(x)
            sigm_a = F.broadcast_to(F.sigmoid(self.a)[:, None, None, None], res.shape)
            return 2 * ((1 - sigm_a) * res + sigm_a * sh)
        else:
            return self.residual(x) + self.shortcut(x)
