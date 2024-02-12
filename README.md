# ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding

[comment]: <> (---)

Official implementation of [ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding](https://arxiv.org/abs/2212.05171)

[comment]: <> (---)

# What is ULIP
ULIP is a Model-agnostic Multimodal Pre-training Framework, which can leverage information from other modalities (Images, Language) to improve the ability to understand 3D data without introducing any extra latency.

[comment]: <> (---)

# Pipeline
![Overall Pipeline](assets/figure2_resize.gif)

[comment]: <> (---)

# Instructions
ULIP is a highly extensible multimodal pre-training framework, and it's model-architecture agnostic, meaning you can easily plug in any 3D backbone models and pre-train it using our framework to get a jump-start for various downstreaming tasks!
## [Install environments]
We pre-train ULIP on 8 Nvidia A100 GPUs, the code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n ulip python=3.7.15``` \
```conda activate ulip``` \
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
# Change the "DATA_PATH", "PC_PATH", "IMAGE_PATH"
./data/ShapeNet-55.yaml
# Change the "DATA_PATH"
./data/ModelNet40.yaml
# Change the initialize_models address
./models/ULIP_models.py
Modify this line "pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))"
```


## [Pre-train 3D backbones]
**Our framework is model architecture agonistic, currently four 3D backbones are supported:** \
**Pointnet2(ssg)**\
**PointBERT**\
**PointMLP**\
**PointNeXt**\
\
Please change the script to accommodate your system accordingly, this script is used to pre-train on 8 gpus by default. You can also modify the desired output folder in the script.
```
# the scripts are named by its correspoinding 3D backbone name.
bash ./scripts/(choose your pre-train script)
```

## [Test pre-trained models for zero-shot classification on ModelNet40]
You may also change the output path in the scripts as well.

```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```
You may also change the output path in the scripts as well.

## [Pre-train & Test using different number of points]
Change the npoints argument in the scripts, by default its 8192. \
**Note: Currently we use FPS to subsample the 8192 points, which might slow down the training speed. If you'd like, you can choose to cache or save the pre-processed datasets with different number of points to speed up your pre-training.**

## [Pre-train your customized 3D backbones]
There are only two things you need to change to pre-train your own customized 3D backbones: \
(1) Define your own 3D backbone in ./models folder.\
We put a template "customized_backbone" here, you can refer to the comments to see the expected input and output shapes. You can also refer to how pointnet2 is defined here. \
(2) Use or modify this "ULIP_CUSTOMIZED" class in ./models/ULIP_models.py.\
Please refer to the comments in "ULIP_CUSTOMIZED" class, it should be straightforward to follow, and please be sure to change the "pc_feat_dims" accordingly (since we are agnostic to the point cloud output feature dimensions of your customized 3D backbones).


# Pre-trained models for zero-shot classification
Zero-shot classification on ModelNet40, 8k points pre-train, 8k points test, best checkpoint:

| model                                                                                                                                                                   | top1 | top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [Pointnet2(ssg)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnet2_ssg.pt?authuser=0) | 57.7 | 78.9 |
| [PointMLP](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointmlp.pt?authuser=0)            | 60.0 | 79.4 |
| [PointBERT](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt?authuser=0)          | 60.3 | 84.0 |
| [PointNeXt](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnext.pt?authuser=0)          | 56.2 | 77.0 |


# Citation

[//]: # (    @article{xue2022ulip,)

[//]: # (      title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},)

[//]: # (      author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},)

[//]: # (      journal={arXiv preprint arXiv:2212.05171},)

[//]: # (      year={2022})

[//]: # (    })
