import shutup

shutup.please()
import os
import argparse
import pathlib
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from data.oxford_paris_datamodule import OxfordParis1MDataModule
from models.matching_module import MatchingTrainingModule
from utils.train_utils import get_training_loggers, get_training_callbacks, prepare_logging_directory


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for homography pretraining')
    parser.add_argument('--config', type=str, help='path to config file', default='config/homography_pretraining.yaml')
    parser.add_argument('--features_config', type=str, help='path to config file with features',
                        default='config/features_online/sift.yaml')
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    features_config = OmegaConf.load(args.features_config)

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # Prepare directory for logs and checkpoints
    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = 'Pretraining_{}__attn_{}__laf_{}__{}'.format(
            features_config['name'],
            config['superglue']['attention_gnn']['attention'],
            config['superglue']['laf_to_sideinfo_method'],
            str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )
        log_path = prepare_logging_directory(config, experiment_name, features_config=features_config)
    else:
        experiment_name, log_path = '', ''

    # Init Lightning Data Module
    data_config = config['data']
    dm = OxfordParis1MDataModule(
        root_path=pathlib.Path(data_config['root_path']),
        batch_size=data_config['batch_size_per_gpu'],
        num_workers=data_config['dataloader_workers_per_gpu'],
        resize_shape=data_config['resize_shape'],
        warp_offset=data_config['warp_offset']
    )

    # Init model
    model = MatchingTrainingModule(
        train_config={**config['train']},
        features_config=features_config,
        superglue_config=config['superglue'],
    )

    # Set callbacks and loggers
    callbacks = get_training_callbacks(config, log_path, experiment_name)
    loggers = get_training_loggers(config, log_path, experiment_name)

    # Init distributed trainer
    trainer = pl.Trainer(
        gpus=config['gpus'],
        max_epochs=config['train']['epochs'],
        accelerator="ddp",
        gradient_clip_val=config['train']['grad_clip'],
        log_every_n_steps=config['logging']['train_logs_steps'],
        limit_train_batches=config['train']['steps_per_epoch'],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=loggers,
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=config['train'].get('precision', 32),
    )
    trainer.fit(model, datamodule=dm, ckpt_path=config.get('checkpoint'))


if __name__ == '__main__':
    main()
