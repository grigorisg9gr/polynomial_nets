import chainer
from chainer import functions as F
from source.links.sn_embed_id import SNEmbedID
from source.links.sn_linear import SNLinear
from dis_models.resblocks_dis import Block, OptimizedBlock


class SNResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=128, n_classes=0, activation=F.relu, sn=False, ch_input=3):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        self.n_classes = n_classes
        with self.init_scope():
            self.block1 = OptimizedBlock(ch_input, ch, sn=sn)
            self.block2 = Block(ch, ch, activation=activation, downsample=True, sn=sn)
            self.block3 = Block(ch, ch, activation=activation, downsample=False, sn=sn)
            self.block4 = Block(ch, ch, activation=activation, downsample=False, sn=sn)
            self.l5 = SNLinear(ch, 1, initialW=chainer.initializers.GlorotUniform(), nobias=True)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch, initialW=chainer.initializers.GlorotUniform())

    def __call__(self, x, y=None, return_feature=False):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = F.sum(h, axis=(2, 3))
        output = self.l5(h)
        if return_feature:
            return output, h
        if y is not None and self.n_classes:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        return output

