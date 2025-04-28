import os
import signal
import sys
import traceback
import hydra

from train_stylegan_real_feature_mlprender import StyleGAN2Trainer


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


@hydra.main(config_path='../config', config_name='stylegan2')
def main(config):

    register_debug_signal_handlers()
    register_quit_signal_handlers()
    config.texopt = True

    if config.mode == 'shapenet':
        config.dataset_path = "/cluster/valinor/abokhovkin/scannet-texturing/data/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color_left"
    else:
        config.dataset_path = "/cluster/daidalos/abokhovkin/scannet-texturing/data/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color" # ScanNet
    config.num_eval_images = 1024

    CHECKPOINT = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/15112113_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr10-5_lrd14-4_lrmd10-4_lrg12-4_lre1-4_noact_stablegs_mlp2x_2gpu/checkpoints/_epoch=70.ckpt"
    CHECKPOINT_EMA = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/15112113_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr10-5_lrd14-4_lrmd10-4_lrg12-4_lre1-4_noact_stablegs_mlp2x_2gpu/checkpoints/ema_000094288.pth"
    CHECKPOINT_EMA_R = "/rhome/abokhovkin/projects/stylegan2-ada-3d-texture/runs/15112113_StyleGAN23D_mlpshader_bigdisc_twodisc_lrr10-5_lrd14-4_lrmd10-4_lrg12-4_lre1-4_noact_stablegs_mlp2x_2gpu/checkpoints/ema_r_000094288.pth"

    model = StyleGAN2Trainer(config).load_from_checkpoint(CHECKPOINT)

    # choose here what you want to store
    # model.store_shapenet_textures()
    # model.store_scannet_textures()
    # model.store_scannet_fornocs()
    model.store_shapenet_fornocs()
    # model.store_rendered_images()


if __name__ == '__main__':
    main()