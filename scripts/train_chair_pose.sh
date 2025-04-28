#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=12
#SBATCH --mem=100gb
##SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate textures

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.3

cd /rhome/abokhovkin/projects/stylegan2-ada-3d-texture

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/textures/bin/python -u trainer/train_chair_pose_classification.py wandb_main=True val_check_interval=5 experiment=chair_scannet_pose_resnet18_lr0.0003 lr=0.0003 sanity_steps=0 image_size=512 batch_size=128
