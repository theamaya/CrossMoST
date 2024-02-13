#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 #,5,6,7,8,9,10,11,12,13,14,15

cd /home/amaya/repos/CrossMoST

set -x

export NCCL_LL_THRESHOLD=0
export MKL_SERVICE_FORCE_INTEL=1

time=`date +%m-%d_%H-%M-%S`

/home/amaya/miniconda3/envs/crossmost/bin/python \
    -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train_CrossMoST_modelnet40.py \
    --output_dir ./outputs/modelnet40_crossmost/ \
    --config ./configs/modelnet40_crossmost.yaml \
    --eval >outputs/modelnet40_crossmost/$time.out
