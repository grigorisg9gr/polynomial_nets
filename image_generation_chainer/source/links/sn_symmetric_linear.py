import chainer
import numpy as np
from chainer.functions.array.broadcast import broadcast_to
from chainer.functions.connection import linear
from chainer.links.connection.linear import Linear
from source.functions.max_sv import max_singular_value


class SNSymmetricLinear(Linear):
    """Symmetric Linear layer with Spectral Normalization.

    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward datasets pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        use_gamma (bool): If true, apply scalar multiplication to the 
            normalized weight (i.e. reparameterize).
        Ip (int): The number of power iteration for calculating the spcetral 
            norm of the weights. 
    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Normalized (Reparametrized) weight parameter.
        b (~chainer.Variable): Bias parameter.
        u (~array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.

    """

    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, Ip=1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNLinear, self).__init__(
            in_size, out_size, nobias, initialW, initial_bias
        )

    @property
    def W_sym(self):
        return self.W + F.transpose(self.W)

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        sigma, _u, _ = max_singular_value(self.W_sym, self.u, self.Ip)
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W_sym.shape)
        self.u = _u
        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W_sym.shape) * self.W_sym / sigma
        else:
            return self.W_sym / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W_sym.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)
