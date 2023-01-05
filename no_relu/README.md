# PiNet without ReLU

<!-- [ALGORITHM] -->

## Abstract

ReLu

<div align=center>
<img src="https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/Top1.png" width="40%"/>
</div>

## Results and models


### ImageNet-1k

|       Model        |  ReLu | Conv1x1 |Top-1 (%) | Top-5 (%) |                                  Backbone                                  |                                  Logs                                   |
| :----------------: | :-------: | :------: | :-------: | :-------: | :----------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
|     ResNet-18      |   Yes  |   No   |   69.90   |   89.43   | [config](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | [log](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.log.json) |
|     ResNet-18      |   No   |   No    |   18.348   |   36.718   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/resnet_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/resnet_norelu.log) |
|     PiNet-18      |   No   |   No   |   63.666   |   84.340   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_norelu.log) |
|     PiNet-18      |   No   |   Yes  |   65.306   |   85.830   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_1x1_norelu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_1x1_norelu.log) |
|     PiNet-18      |   Yes   |   No   |   70.350   |   89.434   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_relu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_relu.log) |
|     PiNet-18      |   Yes   |   Yes   |   71.644   |   90.232   | [backbone](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/backbones/pinet_1x1_relu.py) | [log](https://github.com/grigorisg9gr/polynomial_nets/blob/master/no_relu/logs/pinet_1x1_relu.log) |

