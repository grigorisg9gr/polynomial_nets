import chainer
from chainer import configuration
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable
from chainer.links import EmbedID
import chainer.functions as F


def cond_bn_wrapper(x, c, bn, gamma, beta):
    h = bn(x)
    shape = h.shape
    ndim = len(shape)
    gamma_c = gamma(c)
    beta_c = beta(c)
    gamma_c = F.broadcast_to(F.reshape(gamma_c, list(gamma_c.shape) + [1] * (ndim - 2)), shape)
    beta_c = F.broadcast_to(F.reshape(beta_c, list(beta_c.shape) + [1] * (ndim - 2)), shape)
    return gamma_c * h + beta_c
