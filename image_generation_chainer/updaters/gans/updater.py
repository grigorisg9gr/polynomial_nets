import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
import source.functions.losses as losses
from source.miscs.random_samples import sample_continuous, sample_categorical


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, iter_incndis=None, iter_ndis=1, start_ndisone=False, 
                 mma=None, mma_dis=None, update_gener=True, **kwargs):
        self.mma, self.mma_dis, self.update_gener = mma, mma_dis, update_gener
        self.iter_incndis, self.iter_ndis, self.start_ndisone = iter_incndis, iter_ndis, start_ndisone
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        self.conditional = kwargs.pop('conditional')
        self.n_gen_samples = kwargs.pop('n_gen_samples')
        # # iter_incndis: The iteration(s) to increase n_dis, e.g. in the later stages of the training.
        self.idx_inc, self.ndis_inc = 0, kwargs.pop('ndis_inc') if 'ndis_inc' in kwargs else self.n_dis
        if self.iter_incndis is not None and isinstance(self.iter_incndis, int):
            self.iter_incndis = [(self.iter_incndis, self.ndis_inc)]
        if self.loss_type == 'softplus':
            self.loss_gen = losses.loss_dcgan_gen
            self.loss_dis = losses.loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = losses.loss_hinge_gen
            self.loss_dis = losses.loss_hinge_dis
        elif self.loss_type == 'revKL':
            self.loss_gen = losses.loss_revKL_gen
            self.loss_dis = losses.loss_revKL_dis
        elif self.loss_type == 'wgan':
            self.loss_gen = losses.loss_wgan_gen
            self.loss_dis = losses.loss_wgan_dis

        super(Updater, self).__init__(*args, **kwargs)

    def _generate_samples(self, n_gen_samples=None):
        if n_gen_samples is None:
            n_gen_samples = self.n_gen_samples
        gen = self.models['gen']
        if self.conditional:
            y = sample_categorical(gen.n_classes, n_gen_samples, xp=gen.xp)
        else:
            y = None
        x_fake = gen(n_gen_samples, y=y)
        return x_fake, y

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        x, y = [], []
        for j in range(batchsize):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype(np.int32))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y, dtype=xp.int32)) if self.conditional else None
        return x_real, y_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        if self.start_ndisone:
            if self.iteration == 0:
                # # save the original n_dis and make it 1 in the first few iterations.
                self.org_ndis = int(self.n_dis)
                self.n_dis = 1
                print('Updater (iter={}): Setting n_dis={}.'.format(self.iteration, self.n_dis))
            elif self.iteration == self.iter_ndis:
                # # make the normal n_dis for the rest of the training.
                self.n_dis = int(self.org_ndis)
                # # set the original flag to false to avoid doing the if's.
                self.start_ndisone = False
                print('Updater (iter={}): Setting n_dis={}.'.format(self.iteration, self.n_dis))
        if self.iter_incndis is not None:
            if self.iteration < self.iter_incndis[self.idx_inc][0]:
                self.n_dis = self.iter_incndis[self.idx_inc][1]
            else:
                self.idx_inc += 1
                if len(self.iter_incndis) < self.idx_inc:
                    # # set to None, since we do not have any more parts to change.
                    self.iter_incndis = None
                m1 = 'Updater (iter={}): Was n_dis={} ({}).'
                print(m1.format(self.iteration, self.n_dis, self.iter_incndis))
        for i in range(self.n_dis):
            x_real, y_real = self.get_batch(xp)
            batchsize = len(x_real)
            dis_real = dis(x_real, y=y_real)
            x_fake, y_fake = self._generate_samples(n_gen_samples=batchsize)
            dis_fake = dis(x_fake, y=y_fake)
            x_fake.unchain_backward()

            fake_arr, real_arr = dis_fake.array, dis_real.array
            chainer.reporter.report({'dis_fake': fake_arr.mean()})
            chainer.reporter.report({'dis_real': real_arr.mean()})
            chainer.reporter.report({'fake_max': fake_arr.max()})
            chainer.reporter.report({'real_min': real_arr.min()})

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            loss_dis.unchain_backward()
            chainer.reporter.report({'loss_dis': loss_dis.array})
            del loss_dis

            if i == 0 and self.update_gener:
                x_fake, y_fake = self._generate_samples()
                dis_fake = dis(x_fake, y=y_fake)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                assert not xp.isnan(loss_gen.data)
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                loss_gen.unchain_backward()
                chainer.reporter.report({'loss_gen': loss_gen.array})
                del loss_gen

        self.mma.update(gen) if self.mma is not None else None
