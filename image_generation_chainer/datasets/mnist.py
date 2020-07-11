from os.path import isdir
import chainer
import numpy as np


class MnistDb(chainer.dataset.DatasetMixin):
    def __init__(self, path, ims_suff='train_ims.npy', labels_suff=None, internal_perm=False, 
                 seed=9, permute=False, n_samples=None, train=True, **kwargs):
        # # load the images (loaded all in ram).
        if isdir(path):
            self.base = np.load(path + ims_suff)
        else:
            print('Path not found. Loading fashion-mnist by default.')
            trainset, _ = chainer.datasets.get_fashion_mnist(withlabel=True, scale=1, ndim=3)
            ims = np.array([np.pad(samp[0], ((0, 0), (2, 2), (2, 2)), 'constant') for samp in trainset], dtype=np.float32)
            self.base = ims * 2 - 1

        if self.base[0].ndim == 2:
            # # extend to avoid issues. Channels in front in chainer!!!
            sh1 = self.base.shape
            self.base = np.reshape(self.base, (sh1[0], 1, sh1[1], sh1[2]))
        elif self.base.shape[-1] == 3 and self.base.ndim == 4:
            self.base = self.base.transpose((0, 3, 1, 2))
        if n_samples is not None:
            self.base = self.base[:n_samples]
        # # load the labels if the appropriate npy is provided.
        if labels_suff is not None:
            self.labels = np.load(path + labels_suff)
            if n_samples is not None:
                self.labels = self.labels[:n_samples]
        if internal_perm:
            # # do a random permutation of the data.
            np.random.seed(seed)
            perm = np.random.permutation(range(self.base.shape[0]))
            self.base = self.base[perm]
            if labels_suff is not None:
                self.labels = self.labels[perm]
        self.n_classes = len(set(self.labels)) if (labels_suff is not None) else 0

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image = self.base[i]
        if hasattr(self, 'labels'): 
            return image, self.labels[i]
        else: 
            return image, 1




