import argparse
import numpy as np

from chainer import cuda
from chainer import datasets
from chainer import serializers
from inception_score import Inception
from inception_score import inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--model', type=str, default='inception_score.model')
    return parser.parse_args()


def main(args):
    # Load trained model
    model = Inception()
    serializers.load_hdf5(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Load images
    if 0:
        train, test = datasets.get_cifar10(ndim=3, withlabel=False, scale=255.0)
    else:
        train, test = datasets.get_mnist(ndim=3, rgb_format=True, scale=255.0, withlabel=False)

    # Use all 60000 images, unless the number of samples are specified
    ims = np.concatenate((train, test))
    if args.samples > 0:
        ims = ims[:args.samples]

    mean, std = inception_score(model, ims)

    print('Inception score mean:', mean)
    print('Inception score std:', std)


if __name__ == '__main__':
    args = parse_args()
    main(args)
