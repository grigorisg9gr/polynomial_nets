=======================================
Π-nets: Deep Polynomial Neural Networks
=======================================

Pytorch implementation of the image generation as described on the CVPR'20 paper "**Π-nets: Deep Polynomial Neural Networks**".

Specifically, we include the code for image generation on Cifar10 images.



Browsing the folders
====================
The folder structure is the following:

*    ``data``: The folder that contains the data for Cifar10. If it has not been downloaded and extracted already, it will be downloaded during the first run.

*    ``FID``: The folder that contains the core code for computing the FID scores.

*    ``IS``: The folder that contains the core code for computing the IS scores.


Train the network
=================

First download the following file:

https://www.doc.ic.ac.uk/~sm3515/cifar_10_stats.pkl

We will need this file to calculate the FID score for Cifar10. We have pre-computed the first/second order statistics for Cifar10 database to expedite the training process (so that we do not need to process and extract statistics from the Cifar10 images each time we need to compute the FID score during the validation).

To train the network, you can execute the following command:

   python main.py

The file main.py contains many hyper-parameters which can be either passed as arguments upon the execution of the command above or you can edit the file itself with the hyper-parameters of your liking. If you will be using the default hyper-parameters, the experiments will be saved in the folder dcgan_inject, in the current working directory.



Requirements
============

The code was tested on a Linux machine and requires the following packages/versions to run.

* python 3.5 with cuda toolkit and cuda  10 (if you intend on running on a GPU) 

* tensorflow 1.15

* pytorch 1.2

* torchvision 0.4

* scipy 1.1.0


Snippets of the code are borrowed by the official pytorch DCGAN implementation [1].

The codes for the IS and FID scores are borrowed from [2] and [3], respectively. We should point out that the FID/IS scores that are mentioned in the paper may be slightly different to the ones calculated here since for the paper scores, after having extracted the best network, we run the official Tensorflow FID/IS scores implementations and report those, in order to be consistent throughout the whole set of experiments and frameworks (chainer, pytorch, etc.).

References
==========

.. [1] https://github.com/pytorch/examples/tree/master/dcgan

.. [2] https://github.com/sbarratt/inception-score-pytorch 

.. [3] https://github.com/mseitzer/pytorch-fid
 

