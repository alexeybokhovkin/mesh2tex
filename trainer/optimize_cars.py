import os
import signal
import sys
import traceback
import hydra

from train_stylegan_real_feature_car_deep_twin_progressive_mgf_mlprender import StyleGAN2Trainer


def print_traceback_handler(sig, frame):
    print(f'Received signal {sig}')
    bt = ''.join(traceback.format_stack())
    print(f'Requested stack trace:\n{bt}')


def quit_handler(sig, frame):
    print(f'Received signal {sig}, quitting.')
    sys.exit(1)


def register_debug_signal_handlers(sig=signal.SIGUSR1, handler=print_traceback_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


def register_quit_signal_handlers(sig=signal.SIGUSR2, handler=quit_handler):
    print(f'Setting signal {sig} handler {handler}')
    signal.signal(sig, handler)


@hydra.main(config_path='../config', config_name='stylegan2_car')
def main(config):

    register_debug_signal_handlers()
    register_quit_signal_handlers()
    # if config.mode == 'compcars':
    #     config.texopt = True # CompCars, do not transform mesh vertices to a particular camera view
    # elif config.mode == 'shapenet':
    #     config.texopt = False # ShapeNet, transform mesh vertices to a particular camera view

    if config.task == 'aligned':
        config.texopt = False
    else:
        config.texopt = True

    config.num_eval_images = 1024

    CHECKPOINT = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/22111349_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr5-4_lrmd17-4_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_adamwdec1-2/checkpoints/_epoch=83.ckpt"
    CHECKPOINT_EMA = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/22111349_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr5-4_lrmd17-4_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_adamwdec1-2/checkpoints/ema_000026340.pth"
    CHECKPOINT_EMA_R = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/22111349_StyleGAN23D-CompCars_mlpbig_bigdisc_twodisc_lrr5-4_lrmd17-4_lrd1-3_lrg1-3_lre1-4_mlp2x_2gpu_adamwdec1-2/checkpoints/ema_r_000026340.pth"

    model = StyleGAN2Trainer(config).load_from_checkpoint(CHECKPOINT)

    model.optimize_stylemesh_image_vgg_mlp_aligned(
        ckpt_ema=CHECKPOINT_EMA, 
        ckpt_ema_r=CHECKPOINT_EMA_R, 
        mode=config.mode,
        task=config.task,
        vgg_path=config.vgg_path,
        pose_ckpt_path=config.pose_ckpt_path,
        nocs_dir=config.nocs_dir,
        shapenet_dir=config.shapenet_dir,
        dataset_path=config.dataset_path,
        prefix=config.prefix,
    )


if __name__ == '__main__':
    main()