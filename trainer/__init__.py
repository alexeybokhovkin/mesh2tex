import os
import signal
import sys
import traceback
from pathlib import Path
from random import randint
import datetime

import torch
import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from util.filesystem_logger import FilesystemLogger


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


def generate_experiment_name(name, config):
    if config.resume is not None and not config.create_new_resume:
        experiment = Path(config.resume).parents[1].name
        os.environ['experiment'] = experiment
    elif not os.environ.get('experiment'):
        experiment = f"{datetime.datetime.now().strftime('%d%m%H%M')}_{name}_{config.experiment}"
        os.environ['experiment'] = experiment
    else:
        experiment = os.environ['experiment']
    return experiment


def create_trainer(name, config):
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = generate_experiment_name(name, config)
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)
    # config.seed = 13
    # seed_everything(config.seed, workers=True)

    register_debug_signal_handlers()
    register_quit_signal_handlers()

    if config.resume is not None:
        wandb_resume = True
    else:
        wandb_resume = False
    print('WandB resume:', wandb_resume)

    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'{name}{config.suffix}',
                         name=config.experiment,
                         id=config.experiment,
                         settings=wandb.Settings(start_method='thread'),
                         resume=wandb_resume)
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment / "checkpoints"),
                                          filename='_{epoch}',
                                          save_top_k=-1,
                                          verbose=False,
                                          every_n_epochs=config.save_epoch)

    gpu_count = torch.cuda.device_count()
    print('Num GPUs', gpu_count)

    if gpu_count > 1:

        # config.val_check_interval *= gpu_count
        trainer = Trainer(devices=-1,
                          accelerator='gpu',
                          strategy="ddp",
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    else:
        trainer = Trainer(devices=-1,
                          accelerator='gpu',
                          num_sanity_val_steps=config.sanity_steps,
                          max_epochs=config.max_epoch,
                          limit_val_batches=config.val_check_percent,
                          callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)),
                          check_val_every_n_epoch=max(1, config.val_check_interval),
                          resume_from_checkpoint=config.resume,
                          logger=logger,
                          benchmark=True)
    return trainer
