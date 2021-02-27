=======================================
Π-nets: Deep Polynomial Neural Networks
=======================================

.. image:: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:target: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
	:alt: License

Official implementation of several experiments in the paper `"**Π-nets: Deep Polynomial Neural Networks**" <https://openaccess.thecvf.com/content_CVPR_2020/papers/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.pdf>`_ and its `extension <https://ieeexplore.ieee.org/document/9353253>`_ (also available `here <https://arxiv.org/abs/2006.13026>`_ ).

Each folder contains a different experiment. Please follow the instructions 
in the respective folder on how to run the experiments and reproduce the results. 
`This repository <https://github.com/grigorisg9gr/polynomial_nets>`_ contains implementations in `MXNet <https://mxnet.apache.org/>`_, `PyTorch <https://pytorch.org/>`_ and `Chainer <https://chainer.org/>`_.



Browsing the experiments
========================
The folder structure is the following:

*    ``face_recognition``: The folder contains the code for the `face verification and identification experiments <https://github.com/grigorisg9gr/polynomial_nets/tree/master/face_recognition>`_.

*    ``image_generation_chainer``: The folder  contains the `image generation experiment on Chainer <https://github.com/grigorisg9gr/polynomial_nets/tree/master/image_generation_chainer>`_; specifically the experiment without activation functions between the layers.

*    ``image_generation_pytorch``: The folder contains the `image generation experiment on PyTorch <https://github.com/grigorisg9gr/polynomial_nets/tree/master/image_generation_pytorch>`_; specifically the conversion of a DCGAN-like generator into a polynomial generator.

*    ``mesh_pi_nets``: The folder contains the code for `mesh representation learning <https://github.com/grigorisg9gr/polynomial_nets/tree/master/mesh_pi_nets>`_ with polynomial networks.


More information on Π-nets
==========================


A one-minute pitch of the paper is uploaded `here <https://www.youtube.com/watch?v=5HmFSoU2cOw>`_. We describe there what generation results can be obtained even without activation functions between the layers of the generator. 

Π-nets do not rely on a single architecture, but enable diverse architectures to be built; the architecture is defined by the form of the resursive formula that constructs it. For instance, we visualize below two different Π-net architectures. 

.. image:: figures/model_intro_.png
  :width: 200
  :alt: Different architectures enables by Π-nets.


Results
=======

The evaluation in the paper [1]_ suggests that Π-nets can improve state-of-the-art methods. Below, we visualize results in image generation and errors in mesh representation learning.


.. image:: figures/prodpoly_generation_ffhq.png
  :width: 400
  :alt: Generation results by Π-nets when trained on FFHQ.

The image above shows synthesizes faces. The generator is a Π-net, and more specifically a product of polynomials.


.. image:: figures/dfaust.png
  :width: 400
  :alt: Per vertex reconstruction error on an exemplary human body mesh.

Color coded results of the per vertex reconstruction error on an exemplary human body mesh. From left to right: ground truth mesh, first order SpiralGNN, second, third and fourth order base polynomial in Π-nets. Dark colors depict a larger error; notice that the (upper and lower) limbs have larger error with first order SpiralGNN.



Citing
======
If you use this code, please cite [1]_ or (and) [2]_:

*BibTeX*:: 

  @inproceedings{
  poly2020,
  title={$\Pi-$nets: Deep Polynomial Neural Networks},
  author={Chrysos, Grigorios and Moschoglou, Stylianos and Bouritsas, Giorgos and Panagakis, Yannis and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
  }


*BibTeX*::

  @article{poly2021,
  author={Chrysos, Grigorios and Moschoglou, Stylianos and Bouritsas, Giorgos and Deng, Jiankang and Panagakis, Yannis and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Deep Polynomial Neural Networks}, 
  year={2021},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3058891}}

  
References
==========

.. [1] Grigorios G. Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Yannis Panagakis, Jiankang Deng and Stefanos Zafeiriou, **Π-nets: Deep Polynomial Neural Networks**, *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

.. [2] Grigorios G. Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Jiankang Deng, Yannis Panagakis and Stefanos Zafeiriou, **Deep Polynomial Neural Networks**, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.


