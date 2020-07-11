import copy
import chainer


def copypersistents(src, dst):
    assert isinstance(src, chainer.Link)
    assert isinstance(dst, chainer.Link)

    dst._persistent = list(src._persistent)
    for name in dst._persistent:
        dst.__dict__[name] = copy.deepcopy(src.__dict__[name])

    if isinstance(src, chainer.Chain):
        assert isinstance(dst, chainer.Chain)
        for name in src._children:
            copypersistents(src.__dict__[name], dst.__dict__[name])


def namedpersistents(src):
    assert isinstance(src, chainer.Link)

    for lname, link in src.namedlinks():
        for pname in link._persistent:
            yield lname + '/' + pname, link.__dict__[pname]


def apply_exponential_averaging(src_model, dst_model, alpha):
    src_params = [
        param for _, param in sorted(src_model.namedparams())]
    dst_params = [
        param for _, param in sorted(dst_model.namedparams())]
    assert (len(src_params) == len(dst_params))
    for p, avg_p in zip(src_params, dst_params):
        avg_p.data[:] += float(1 - alpha) * (p.data - avg_p.data)

    src_persistents = [
        param for _, param in sorted(namedpersistents(src_model))]
    dst_persistents = [
        param for _, param in sorted(namedpersistents(dst_model))]
    assert (len(src_persistents) == len(dst_persistents))
    for src, dst in zip(src_persistents, dst_persistents):
        dst *= alpha
        dst += src * (1 - alpha)


class ModelMovingAverage(object):
    name = "ModelMovingAverage"

    def __init__(self, alpha, model=None):
        self.alpha = alpha
        self.avg_model = copy.deepcopy(model) if model is not None else None

    def update(self, src_model):
        if self.avg_model is None:
            self.avg_model = copy.deepcopy(src_model)
        apply_exponential_averaging(src_model, self.avg_model, self.alpha)

    def copy_avg_model_to(self, model):
        if self.avg_model is None:
            raise Exception("self.avg_model is None")
        apply_exponential_averaging(self.avg_model, model, 0)

    def set_avg_model(self, model):
        self.avg_model = copy.deepcopy(model)
