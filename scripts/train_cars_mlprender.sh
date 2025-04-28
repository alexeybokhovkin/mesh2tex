#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexey.bokhovkin@skoltech.ru
#SBATCH --cpus-per-task=12
#SBATCH --mem=160gb
##SBATCH --constraint="rtx_a6000"
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,gimli,balrog,andram
#SBATCH -p submit

export PYTHONPATH=.
export PATH=/rhome/${USER}/miniconda3/bin/:$PATH
source activate textures

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/11.3

cd /rhome/abokhovkin/projects/stylegan2-ada-3d-texture

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1 /rhome/abokhovkin/miniconda3/envs/textures/bin/python -u trainer/train_stylegan_real_feature_car_deep_twin_progressive_mgf_mlprender.py wandb_main=True val_check_interval=2 experiment=mlpbig_bigdisc_twodisc_lrr5-4_lrmd55-5_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_noglobstyle lr_r=0.0005 lr_mlp_d=0.00055 lr_g=0.0015 lr_d=0.0017 lr_e=0.0001 sanity_steps=0 lambda_gp=14 image_size=512 render_size=384 batch_size=2 num_mapping_layers=8 views_per_sample=8 g_channel_base=32768 g_channel_max=768 random_bg=grayscale num_vis_images=256 num_eval_images=512 preload=False progressive_switch=[80e3,160e3] alpha_switch=[80e3,160e3] progressive_start_res=7 features=normal num_faces=[24576,6144,1536,384,96,24] load_checkpoint=False freeze_texturify=False small_shader=False accumulate_grad_batches=1 overfit=False tanh_render=True use_act_render=False create_new_resume=False residual=False render_shape_features=True
