import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from source.links.sn_linear import SNLinear
from source.links.sn_convolution_2d import SNConvolution2D
from source.links.categorical_conditional_batch_normalization import CategoricalConditionalBatchNormalization
try:
    from source.links.instance_normalization import InstanceNormalization
except ImportError:
    #print('The instance norm is not implemented or imported in the generator.')
    pass
from functools import partial

from source.miscs.random_samples import sample_continuous, sample_categorical


def return_norm(norm):
    if isinstance(norm, str):
        return norm
    if norm == 0:
        return 'batch'
    elif norm == 1:
        return 'instance'
    elif norm == 2:
        return 'cinstance'


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


class ProdPolyConvGenerator(chainer.Chain):
    def __init__(self, layer_d, use_bn=False, sn=False, out_ch=2, mult_lat=True,
                 distribution='uniform', ksizes=None, strides=None, paddings=None, 
                 activ=None, bottom_width=1, use_localz=True, n_classes=0, 
                 use_out_act=False, channels=None, power_poly=[4, 3], 
                 derivfc=0, activ_prod=0, use_bias=True, dim_z=128, train=True,
                 add_h_injection=False, add_h_poly=False, norm_after_poly=False,
                 normalize_preinject=False, use_act_zgl=False, allow_conv=False,
                 repeat_poly=1, share_first=False, type_norm='batch', typen_poly='batch',
                 skip_rep=False, order_out_poly=None, use_activ=False, thresh_skip=None,
                 use_sec_recurs=False, sign_srec=-1):
        """
        Initializes the product polynomial generator.
        :param layer_d: List of all layers' inputs/outputs channels (including 
                        the input to the last output).
        :param use_bn: Bool, whether to use batch normalization.
        :param sn: Bool, if True use spectral normalization.
        :param out_ch: int, Output channels of the generator.
        :param ksizes: List of the kernel sizes per layer.
        :param strides: List of the stride per layer.
        :param paddings: List of the padding per layer (int).
        :param activ: str or None: The activation to use.
        :param bottom_width: int, spatial resolution to reshape the init noise.
        :param use_localz: bool, if True, then use 'local' transformations (affine)
               to transform from the original z to the current shape.
        :param use_out_act: bool, if True, it uses a tanh as output activation.
        :param repeat_poly: int or list: List of how many times to repeat each 
               polynomial (original with FC layers, the last is NOT repeated).
        :param share_first: bool. If True, it shares the first FC layer in the
               FC polynomials. It assumes the respective FC layers are of the same size.
        :param type_norm: str. The type of normalization to be used, i.e. 'batch' for
               BN, 'instance' for instance norm.
        :param typen_poly: str. The type of normalization to be used ONLY 
               for the FC layers. 
        :param skip_rep: bool. If True, it uses a skip connection before the layer to 
               the next, i.e. in practice it directly skips the convolution/fc plus hadamard 
               product of this layer. See derivation 3.
        :param order_out_poly: int or None. If int, a polynomial of the order is used
               in the output, i.e. before the final convolution.
        :param use_activ: bool. If True, use activations in the main deconvolutional part.
        :param thresh_skip: None, int or list. It represents the probability of skipping the
               hadamard product in the deconvolution part (i.e. main polynomial). This is per
               layer; e.g. if 0.3, this means that approx. 30% of the time this hadamard will
               be skipped. It should have values in the [0, 1]. In inference, it should be
               deactivated.
        :param use_sec_recurs: bool, default False. If True, it uses the second recursive term, 
               i.e. x_(n+1)=f(x_n, x_(n-1)). This is used only in the derivation 3 setting. Also, 
               for practical reasons, use only for the FC layers when their order > 3.
        """
        super(ProdPolyConvGenerator, self).__init__()
        self.n_l = n_l = len(layer_d) - 1
        w = chainer.initializers.GlorotUniform()
        if sn:
            Conv = SNConvolution2D
            Linear = SNLinear
            raise RuntimeError('Not implemented!')
        else:
            Conv = L.Deconvolution2D
            Linear = L.Linear
        Conv0 = L.Convolution2D
        # # initialize args not provided.
        if ksizes is None:
            ksizes = [4] * n_l
        if strides is None:
            strides = [2] * n_l
        if paddings is None:
            paddings = [1] * n_l
        # # the length of the ksizes and the strides is only for the conv layers, hence
        # # it should be one number short of the layer_d.
        assert len(ksizes) == len(strides) == len(paddings) == n_l
        # # save in self, several useful properties.
        self.use_bn = use_bn
        self.n_channels = layer_d
        self.mult_lat = mult_lat
        self.distribution = distribution
        # # Define dim_z.
        assert isinstance(dim_z, int)
        self.dim_z = dim_z
        self.train = train
        # # bottom_width: The default starting spatial resolution of the convolutions.
        self.bottom_width = bottom_width
        self.use_localz = use_localz
        self.n_classes = n_classes
        if activ is not None:
            activ = getattr(F, activ)
        elif use_activ:
            activ = F.relu
        else:
            activ = lambda x: x
        self.activ = self.activation = activ
        self.out_act = F.tanh if use_out_act else lambda x: x
        # # Set the add_one to True out of consistency with the other 
        # # implementation (resnet-based).
        self.add_one = 1
        self.add_h_injection = add_h_injection
        self.normalize_preinject = normalize_preinject
        self.use_bias = use_bias
        self.add_h_poly = add_h_poly
        self.power_poly = power_poly
        if channels is None:
            # # Initialize in the dimension of z.
            channels = [dim_z] * len(self.power_poly)
        self.channels = channels
        if isinstance(activ_prod, int):
            activ_prod = [activ_prod] * len(self.power_poly)
        self.activ_prod = activ_prod
        if isinstance(repeat_poly, int):
            repeat_poly = [repeat_poly] * len(self.power_poly)
        self.repeat_poly = repeat_poly
        assert len(self.power_poly) == len(self.activ_prod) == len(self.channels)
        assert len(self.power_poly) == len(self.repeat_poly)
        # # If True, it compensates for python open interval in the end of range. 
        self.derivfc = derivfc
        self.norm_after_poly = norm_after_poly
        # # whether to use an activation in the global transformation.
        self.use_act_zgl = use_act_zgl and use_activ
        self.share_first = share_first
        self.type_norm = return_norm(type_norm)
        self.typen_poly = return_norm(typen_poly)
        self.skip_rep = skip_rep
        # # optionally adding one 'output polynomial', before making the final output convolution.
        self.order_out_poly = order_out_poly
        # # set the threshold for skipping the injection in the deconvolution part.
        if isinstance(thresh_skip, float) and self.train:
            self.thresh_skip = [0] + [thresh_skip] * (n_l - 1)
            print('[Gen] Skip probabilities: {}.'.format(self.thresh_skip))
        else:
            self.thresh_skip = thresh_skip
        self.use_sec_recurs = use_sec_recurs and skip_rep
        self.sign_srec = sign_srec

        with self.init_scope():
            if self.n_classes == 0:
                bn1 = partial(L.BatchNormalization, use_gamma=True, use_beta=False)
            elif type_norm == 'batch':
                print('[Gen] Categorical batch normalization in the generator.')
                bn1 = partial(CategoricalConditionalBatchNormalization, n_cat=n_classes)
            elif type_norm == 'instance':
                print('[Gen] Instance normalization in the generator.')
                bn1 = InstanceNormalization
            if self.typen_poly != self.type_norm and self.typen_poly == 'instance':
                print('[Gen] Using instance normalization for the FC layers *ONLY*.')
                bn2 = InstanceNormalization
            else:
                bn2 = bn1
            # # inpc_pol: the input size to the current polynomial; initialize on dimz.
            inpc_pol = dim_z
            # # iterate over all the polynomials (in the fully-connected part).
            print('[Gen] Number of product polynomials: {}.'.format(len(self.power_poly)))
            for id_poly in range(len(self.power_poly)):
                chp = self.channels[id_poly]
                m1 = '[Gen] Channels of the polynomial {}: {}, depth: {}.'
                print(m1.format(id_poly, chp, self.power_poly[id_poly]))
                # ensure that the current input channels match the expected.
                setattr(self, 'has_rsz{}'.format(id_poly), inpc_pol != chp)
                if inpc_pol != chp:
                    setattr(self, 'resize{}'.format(id_poly), Linear(inpc_pol, chp, nobias=not use_bias, initialW=w))
                # # now build the current polynomial (id_poly).
                for l in range(1, self.power_poly[id_poly] + self.add_one):
                    if l == 1 and id_poly > 0 and self.share_first:
                        # # in this case the layer will be shared with the first polynomial's
                        # # the first layer.
                        continue
                    setattr(self, 'l{}_{}'.format(id_poly, l), Linear(chp, chp, nobias=not use_bias, initialW=w))
                # # define the activation for this polynomial.
                actp = F.relu if self.activ_prod[id_poly] else lambda x: x
                setattr(self, 'activ{}'.format(id_poly), actp)
                if self.norm_after_poly:
                    # # define different batch normalizations for each polynomial repeated.
                    for repeat in range(self.repeat_poly[id_poly]):
                        setattr(self, 'bnp{}_r{}'.format(id_poly, repeat), bn2(chp))
                # # update input for the next polynomial.
                inpc_pol = int(chp)

            if bottom_width > 1:
                # # make a linear layer to transform to this shape.
                setattr(self, 'lin0', Linear(inpc_pol, inpc_pol * bottom_width ** 2, initialW=w))

            # # iterate over all layers (till the last) and save in self.
            for l in range(1, n_l + 1):
                # # define the input and the output names.
                ni, no = layer_d[l - 1], layer_d[l]
                Conv_sel = Conv0 if strides[l - 1] == 1 and ksizes[l - 1] == 3 and allow_conv else Conv
                conv_i = partial(Conv_sel, initialW=w, ksize=ksizes[l - 1], 
                                 stride=strides[l - 1], pad=paddings[l - 1])
                # # save the self.layer.
                setattr(self, 'l{}'.format(l), conv_i(ni, no))
                if self.skip_rep:
                    # # define some 1x1 convolutions iff the channels are not the same in
                    # # the input and the output. Otherwise, use the identity mapping.
                    func = (lambda x: x) if ni == no else Conv(ni, no, ksize=1)
                    setattr(self, 'skipch{}'.format(l), func)

            if self.order_out_poly is not None:
                ch1 = layer_d[n_l]
                for i in range(1, self.order_out_poly):
                    setattr(self, 'oro{}'.format(i + 1), Conv0(ch1, ch1, ksize=3, stride=1, pad=1, initialW=w))

            # # save the last layer.
            # # In Deconv, we need to define a ksize (otherwise the dimensions unchanged). This 
            # # last layer leaves the spatial dimensions untouched.
            setattr(self, 'l{}'.format(n_l + 1), Conv(layer_d[n_l], out_ch, ksize=3, 
                                                      pad=1, initialW=w))
            self.n_channels.append(out_ch)
            if use_bn:
                # # set the batch norm (applied before the first layer conv).
                setattr(self, 'bn{}'.format(1), bn1(layer_d[0]))
                for l in range(2, self.n_l + 1):
                    # # set the batch norm for the layer.
                    setattr(self, 'bn{}'.format(l), bn1(layer_d[l - 1]))
            if self.mult_lat and use_localz:
                # # define the 'local' transformations of z (z local Linear).
                for l in range(1, n_l + 1):
                    ni, no = layer_d[0], layer_d[l]
                    setattr(self, 'locz{}'.format(l), Linear(ni, no, initialW=w))
            # # the condition to add a residual skip to the representation.
            self.skip1 = self.add_h_injection or self.add_h_poly or self.skip_rep

    def return_injected(self, h, z, n_layer, mult_until_exec=None):
        """ Performs the Hadamard products with z. """
        # # check whether to skip the hadamard.
        skip_injection = False 
        if self.thresh_skip is not None and self.thresh_skip[n_layer-1] > 0:
            # # skip the hadamard, iff the random number is smaller than the threshold.
            skip_injection = np.random.uniform() < self.thresh_skip[n_layer-1]
        if not skip_injection and mult_until_exec is not None:
            skip_injection = mult_until_exec <= n_layer
        if self.mult_lat and not skip_injection:
            if self.use_localz:
                # # apply local transformation.
                z1 = getattr(self, 'locz{}'.format(n_layer))(z)
            else:
                z1 = z
            # # appropriately reshape z for the elementwise multiplication.
            sh = h.shape
            z1 = F.reshape(z1, (sh[0], sh[1], 1))
            if self.normalize_preinject:
                z1 /= F.sqrt(F.mean(z1 * z1, axis=1, keepdims=True) + 1e-8)
            z2 = F.repeat(z1, sh[3] * sh[2], axis=2)
            z2 = F.reshape(z2, sh)
            ret = h * z2 + h if self.add_h_injection else h * z2
            return ret
        return h

    def prod_poly_FC(self, z, mult_until_exec, batchsize=None, y=None):
        """
        Performs the products of polynomials for the fully-connected part.
        """
        # # input_poly: the input variable to each polynomial; for the 
        # # first, simply z, i.e. the noise vector.
        input_poly = z + 0
        # # iterate over all the polynomials (length of channels many).
        for id_poly in range(len(self.channels)):
            # # ensure that the channels from previous polynomial are of the
            # # appropriate size.
            if getattr(self, 'has_rsz{}'.format(id_poly)):
                input_poly = getattr(self, 'resize{}'.format(id_poly))(input_poly)

            # # id_first: The index of the first FC layer; if we share it, it should
            # # be that of l0_1; otherwise l[id_poly]_1.
            id_first = id_poly if not self.share_first else 0
            # # order of the current polynomial.
            order = self.power_poly[id_poly] + self.add_one
            # # condition to use the second recursive term.
            sec_rec = order > 2 and self.skip1 and self.use_sec_recurs
            # # Repeat each polynomial repeat_poly times (if repeat==1, then only running once).
            for repeat in range(self.repeat_poly[id_poly]):
                h = getattr(self, 'l{}_1'.format(id_first))(input_poly)
                if sec_rec:
                    all_reps = []

                # # loop over the current polynomial layers and compute the 
                # # output (for this polynomial). 
                for layer in range(2, order):
                    if layer <= mult_until_exec:
                        # # step 1: perform the hadamard product.
                        if self.derivfc == 0:
                            z1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(input_poly)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += z1 * h
                            else:
                                h = z1 * h 
                        elif self.derivfc == 1:
                            # # In this case, we assume that both A_i, B_i of
                            # # the original polygan derivation are identity matrices.
                            h1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(h)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += h1 * input_poly
                            else:
                                h = h1 * input_poly
                        if sec_rec and layer > 2:
                            h = h + self.sign_srec * all_reps[-2]
                    # # step 2: activation.
                    h = getattr(self, 'activ{}'.format(id_poly))(h)
                # # if we included the second recursive term, add back in the final representation.
                if sec_rec:
                    for rep in all_reps:
                        h = h - self.sign_srec * rep
                if self.norm_after_poly:
                    # # apply a different BN for every repeat time of the poly.
                    if y is None or self.typen_poly == 'instance':
                        h = getattr(self, 'bnp{}_r{}'.format(id_poly, repeat))(h)
                    else:
                        h = getattr(self, 'bnp{}_r{}'.format(id_poly, repeat))(h, y)
                # # update the input for the next polynomial.
                input_poly = h + 0

        # # use the output of the products above as z.
        z = input_poly if not self.use_act_zgl else self.activation(input_poly)
        return z

    def __call__(self, batchsize=None, y=None, z=None, mult_until_exec=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        activ = self.activation

        # # mult_until_exec: If set, we perform the multiplications until that layer.
        # # In the product of polynomials, it applies the same rule for *every*
        # # polynomial. E.g. if mult_until_exec == 2 it will perform the hadamard
        # # products until second order terms in every polynomial.
        if mult_until_exec is None:
            mult_until_exec = 10000
        z = self.prod_poly_FC(z, mult_until_exec, batchsize=batchsize, y=y)

        h = z + 0
        if self.bottom_width > 1:
            h = getattr(self, 'lin0')(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))

        # # loop over the layers and get the layers along with the
        # # normalizations per layer.
        for l in range(1, self.n_l + 1):
            if self.skip_rep:
                h_hold = h + 0
            if self.use_bn and y is None:
                h = getattr(self, 'bn{}'.format(l))(h)
            elif self.use_bn:
                h = getattr(self, 'bn{}'.format(l))(h, y)
            h = activ(getattr(self, 'l{}'.format(l))(h))
            h = self.return_injected(h, z, l, mult_until_exec=mult_until_exec)
            if self.skip_rep:
                # # transform the channels of h_hold if required.
                h_hold = getattr(self, 'skipch{}'.format(l))(h_hold)
                # # upsample if required.
                if h_hold.shape[-1] != h.shape[-1]:
                    h_hold = _upsample(h_hold)
                h += h_hold
        if self.order_out_poly is not None:
            z0 = h + 0
            for i in range(1, self.order_out_poly):
                h1 = getattr(self, 'oro{}'.format(i + 1))(h)
                if self.skip_rep:
                    # # model3 polynomial.
                    h += z0 * h1
                else:
                    h = z0 * h1
        # # last layer (no activation).
        output = getattr(self, 'l{}'.format(self.n_l + 1))(h)
        out = self.out_act(output)
        return out

    def __str__(self, **kwargs):
        m1 = 'Layers: {}.\t Info for channels: {}.'
        str1 = m1.format(self.n_l, self.n_channels)
        return str1

