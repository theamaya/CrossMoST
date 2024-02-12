# CrossMoST: Cross-Modal Self-Training: Aligning Images and Pointclouds to learn Classification without Labels

[comment]: <> (---)

Official implementation of [Cross-Modal Self-Training: Aligning Images and Point Clouds to learn Classification without Labels](https://arxiv.org/)

[comment]: <> (---)

# What is CrossMoST
It is an optimization framework to improve the label-free classification performance of a zero-shot 3D vision model by leveraging unlabeled 3D data and their accompanying 2D views. 
We implement a student-teacher framework to simultaneously process 2D views and 3D point clouds and generate joint pseudo labels to train a classifier and guide cross-model feature alignment.
                                                                                                                        
[comment]: <> (---)

# Pipeline
![Overall Pipeline](Assets/CrossMoST.pdf)

[comment]: <> (---)

# Instructions
## [Install environments]
We trained our models on 4 Nvidia V100 GPUs, the code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n crossmost python=3.7.15``` \
```conda activate crossmost``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```

## [Download datasets and initialize models, put them in the right paths.]
Download the used datasets and initialize models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "initialize_models", "modelnet40_normal_resampled", and "shapenet-55". You might need a gmail account to access it.\
After you download the datasets and initialize models, you can choose one of the following options: \
(1) Put it in or do a soft link to the data folder, by default the data folder should have the following structure:
```
./data |
-- ModelNet40.yaml |
-- ShapeNet-55.yaml |
-- dataset_3d.py |
-- dataset_catalog.json |
-- initialize_models |
-- labels.json |
-- modelnet40_normal_resampled |
-- shapenet-55 |
-- templates.json
```
(2) Change the paths accordingly (optional to do if you don't want to put/link downloaded files in the data folder):
```
# Change the data paths in the config files
./data/[dataset].yaml
```

## [Zero-shot evaluation of Shapenet-pretrained backbones]
ULIP pretraining??

Please change the script to accommodate your system accordingly, this script is used to train on 4 gpus by default. You can also modify the desired output folder in the script.
```
# the scripts are named by its correspoinding 3D backbone name.
bash ./scripts/(choose your pre-train script)
```

## [Training CrossMoST]
You may also change the output path in the scripts as well.
```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```
You can also run the baseline-self training
```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```

# Checkpoints for evaluating Baseline Self-training vs CrossMoST 
Zero-shot classification on ModelNet40, 8k points pre-train, 8k points test, best checkpoint:

| model                                                                                                                                                                   | top1 | top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [Pointnet2(ssg)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnet2_ssg.pt?authuser=0) | 57.7 | 78.9 |
| [PointMLP](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointmlp.pt?authuser=0)            | 60.0 | 79.4 |
| [PointBERT](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt?authuser=0)          | 60.3 | 84.0 |
| [PointNeXt](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnext.pt?authuser=0)          | 56.2 | 77.0 |

Evaluating CrossMoST using the given checkpoints
```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```
You can also run the evaluations for baseline-self training
```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```

# Our repository is based on 
ULIP

# Citation

[//]: # (    @article{xue2022ulip,)

[//]: # (      title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},)

[//]: # (      author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},)

[//]: # (      journal={arXiv preprint arXiv:2212.05171},)

[//]: # (      year={2022})

[//]: # (    })

