Official Implementation of the experiments on mesh representation learning with polynomial networks as described in the CVPR'20 paper [Π-nets: Deep Polynomial Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.pdf) and its [extension](https://arxiv.org/abs/2006.13026).

This repository is based on Neural3DMM [[paper](https://arxiv.org/abs/1905.02876), [code](https://github.com/gbouritsas/Neural3DMM)]. Please refer to the Neural3DMM repository for technical details (e.g. regarding the local vertex orderings or the mesh downsampling operations) and for explanations of the data organisation/folder hierarchy.


## Data

We provide the templates (mesh topology) as well as their downsampled versions for the two datasets that we experimented with, namely [COMA](https://coma.is.tue.mpg.de/) and [DFAUST](http://dfaust.is.tue.mpg.de/). Unfortunately, we cannot provide the data in this repository, but they can be easily obtained from the authors of the datasets. Then, follow the instructions in the [Neural3DMM repo](https://github.com/gbouritsas/Neural3DMM) to organise the data as expected from the code. For COMA we use the splits provided by the authors (in the paper we report the results on the 'sliced' version of the dataset - interpolation experiment). If you would like to train on the same splits for DFAUST, please contact us.

## Important parameters

```
  --injection: if set to True runs the polynomial version of SpiralNet. Alternatively, plain SpiralNet is created.
  --order: the order of the polynomial
  --model:'full' creates the polynomial that NCP decomposition yields, 
          'simple' creates the parameter efficient polynomial (element-wise multiplications akin to polynomial activation functions)
  --normalize: 'final' (only the highest order term is normalised),
                '2nd' (normalises the 2nd order term of each iterate of the NCP recursive formulation), 
                'all' (normalises the output of each iterate of the NCP recursive formulation)
  --residual: if True, it invokes the NCP-skip decomposition
  --activation: if set to identity, no activation functions are used (linear w.r.t. weights)
```

## Training

Example usage:

**COMA**
```
python spiral_pi_nets.py --root_dir ./datasets --name sliced --dataset COMA --mode train --order 3 --model full  --normalize 2nd --activation identity --residual True --results_folder 3rd_order_full_norm_2nd_residual_linear --device_idx 0 --batch_size 16 
```

**DFAUST**
```
python spiral_pi_nets.py --root_dir ./datasets --dataset DFAUST --mode train --order 3 --model full  --normalize 2nd --activation identity --residual True --results_folder 3rd_order_full_norm_2nd_residual_linear --device_idx 0 --batch_size 16 
```

## Testing

Assign to the _results_folder_ argument the value of the folder where the pretrained network is saved. By setting the _mm_constant_ you can obtain measurements in milimiters. For COMA and DFAUST this value is 10^3. Example usage:

**COMA**
```
python spiral_pi_nets.py --root_dir ./datasets --name sliced --dataset COMA --mode test --order 3 --model full  --normalize 2nd --activation identity --residual True --results_folder 3rd_order_full_norm_2nd_residual_linear --device_idx 0 --batch_size 16 --mm_constant 1000
```

**DFAUST**
```
python spiral_pi_nets.py --root_dir ./datasets --dataset DFAUST --mode test --order 3 --model full  --normalize 2nd --activation identity --residual True --results_folder 3rd_order_full_norm_2nd_residual_linear --device_idx 0 --batch_size 16 --mm_constant 1000
```


## Dependencies and Installation Instructions

Requirements:
- python 3.7
- pytorch>=1.2.0
- cudatoolkit>=9.2
- scipy
- trimesh
- sklearn
- tqdm
- tensorboardX

Recommended installation instructions
```
conda create --name pi_spirals python=3.7
conda activate pi_spirals
conda install pytorch==1.2.0 cudatoolkit=9.2 -c pytorch
pip install scipy
pip install trimesh
pip install sklearn
pip install tqdm
pip install tensorboardX
```

