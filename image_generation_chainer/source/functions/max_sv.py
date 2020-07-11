import chainer
from chainer import cuda
from chainer.functions.math import sum
from chainer.functions.connection import linear
from chainer.functions.array import transpose
import numpy as np


def _l2normalize_gpu(v, eps=1e-12):
    norm = cuda.reduce('T x', 'T out',
                       'x * x', 'a + b', 'out = sqrt(a)', 0,
                       'norm_sn')
    div = cuda.elementwise('T x, T norm, T eps',
                           'T out',
                           'out = x / (norm + eps)',
                           'div_sn')
    return div(v, norm(v), eps)


def _l2normalize_cpu(v, eps=1e-12):
   return v / (((v ** 2).sum()) ** 0.5 + eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    if xp == np:
        _l2normalize = _l2normalize_cpu
    else:
        _l2normalize = _l2normalize_gpu
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = sum.sum(linear.linear(_u, transpose.transpose(W)) * _v)
    return sigma, _u, _v
