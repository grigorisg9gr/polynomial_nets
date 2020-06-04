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

4. Comparing with the [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition) code. We only add three lines between line 409 and 412 of the symbol/fresnet.py:

```Shell
act2 = Act(data=bn3*shortcut, act_type='tanh', name=name + '_tanh1')
conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1,1), pad=(0, 0),
             no_bias=True, workspace=workspace, name=name + '_conv3')
bn4 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn4')
```
5. Pre-trained models can be downloaded from [here](https://www.dropbox.com/sh/0kh42qinncf73q9/AAA9J2mAewa48P-xXsIPOAdia?dl=0):

### Citation

If you find this code useful in your research, please consider to cite the following related papers:

```
@inproceedings{chrysos2020pi,
  title={$$\backslash$Pi-$ nets: Deep Polynomial Neural Networks},
  author={Chrysos, Grigorios G and Moschoglou, Stylianos and Bouritsas, Giorgos and Deng, Jiankang and Panagakis, Yannis and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={CVPR},
  year={2019}
}
```
