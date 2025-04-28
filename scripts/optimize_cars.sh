#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
##SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,gimli,balrog
##SBATCH --constraint="rtx_2080"
#SBATCH -p submit
##SBATCH --output=UnAlignCompCarsOurs-%x.%j.out

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate textures

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.3

cd /rhome/abokhovkin/projects/stylegan2-ada-3d-texture

HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/textures/bin/python -u trainer/optimize_cars.py wandb_main=True val_check_interval=1 experiment=zsplit2_mespassback lr_d=0.001 sanity_steps=0 lambda_gp=14 image_size=512 render_size=384 batch_size=1 num_mapping_layers=8 views_per_sample=1 g_channel_base=32768 g_channel_max=768 random_bg=grayscale num_vis_images=256 preload=False features=normal latent_dim=64 postfix=15 num_faces=[24576,6144,1536,384,96,24] load_checkpoint=False freeze_texturify=False small_shader=False accumulate_grad_batches=1 overfit=False tanh_render=True use_act_render=False create_new_resume=False residual=False render_shape_features=True mode=compcars
