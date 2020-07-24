Official Implementation of the experiments on discriminative tasks with polynomial networks as described in the CVPR'20 paper [Π-nets: Deep Polynomial Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.pdf) and its [extension](https://arxiv.org/abs/2006.13026).

### Model Training

1. Download the training dataset [`MS1MV3-RetinaFace`](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/).
The training dataset includes the following 6 files:

```Shell
    ms1m-retinaface/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.

2. Edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

3. Our experiments were conducted on the 8*2080ti GPU. The training script is like this:

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_parall.py --network r50 --loss arcface --dataset retina > log_ms1mretina_ploynet50_arcface.txt  2>&1 
```

4. Compared with the [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition) code, we only add three lines between line 409 and 412 of the symbol/fresnet.py:

```Shell
act2 = Act(data=bn3*shortcut, act_type='tanh', name=name + '_tanh1')
conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1,1), pad=(0, 0),
             no_bias=True, workspace=workspace, name=name + '_conv3')
bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
```
5. Pre-trained models can be downloaded from [here](https://www.dropbox.com/sh/0kh42qinncf73q9/AAA9J2mAewa48P-xXsIPOAdia?dl=0).

6. Results on [IJB-C](https://github.com/deepinsight/insightface/tree/master/Evaluation/IJB):

| Method    | TAR@FAR=1e-5 | TAR@FAR=1e-4 |
| -------   | ------       | --------- | 
|ResNet50   | 94.28        | 96.17     |  
|Prodpoly-ResNet50   | 95.16 | 96.58     | 


### Dependencies
Requirements:
- python 3.x
- mxnet
- opencv


