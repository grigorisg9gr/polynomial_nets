# PiNet without Activation Function

<!-- [ALGORITHM] -->

## Abstract

Typically, neural networks required activation functions to be able to approximate effectively complex distributions. ReLU-nets have been popular, e.g., ResNets. However, in the proposed polynomial nets, there is no strict requirement for activation functions as the Π-nets already include nonlinear interactions between the input elements. In fact, you could capture high-order correlations between the input elements without any activation functions, which is what we focus on in this experiment. In particular, we illustrate how Π-nets can learn classification even in the demanding ImageNet without activation functions. We hope that our code can inspire further experimentation with networks that do not require activation functions and can find alternative ways to express nonlinear relationships between the input elements. 



<div align=center>
<img src="https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/Top1.png" width="40%"/>
  <img src="https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/Top5.png" width="40%"/>
</div>

## Implemenation Details

Please follow [mmclassification](https://github.com/open-mmlab/mmclassification) to set up the training environment. Our models are trained by a single server with eight V100 GPUs.

We slightly modifiy [ResNet](https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/backbones/resnet.py) for different [experiments](https://github.com/grigorisg9gr/polynomial_nets/tree/master/no_relu/backbones).

All other training details follow the standard [configuration](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py).

## Results 

### ImageNet-1k

|       Model        |  ReLu | Conv-1x1 |Top-1 (%) | Top-5 (%) |                                  Backbone                                  |                                  Logs                                   |
| :----------------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
|     ResNet-18      |   Yes  |   No   |   69.90   |   89.43   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json) |
|     ResNet-18      |   No   |   No    |   18.348   |   36.718   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/resnet_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/resnet_norelu.log) |
|     PiNet-18      |   No   |   No   |   63.666   |   84.340   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_norelu.log) |
|     PiNet-18      |   No   |   Yes  |   65.306   |   85.830   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_1x1_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_1x1_norelu.log) |
|     PiNet-18      |   Yes   |   No   |   70.350   |   89.434   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_relu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_relu.log) |
|     PiNet-18      |   Yes   |   Yes   |   71.644   |   90.232   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_1x1_relu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_1x1_relu.log) |

