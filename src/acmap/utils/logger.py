import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import wandb
from acmap.utils.config import Config


class Logger:
    def __init__(self, config: Config, seed, make_ckpts=True):
        self.config = config
        self.prefix = f'{config.prefix}-' if config.prefix != '' else config.prefix
        self.run_group = os.path.join(config.exp.dataset, f'b{config.init_cls}-inc{config.increment}')

        self.run_name = self.prefix
        self.run_name += config.exp.name
        self.run_name += f'-{config.our.merge_method}'
        self.run_name += '-in21k' if '_in21k' in config.exp.backbone_type else ''
        self.run_name += f'-seed{seed}'
        self.run_name += f'-debug-{datetime.now().strftime("%Y%m%d-%H:%M:%S")}' if config.debug else ''

        self.ckpts_dir = os.path.join(config.ckpts_dir, self.run_group, self.run_name)

        if make_ckpts:
            os.makedirs(self.ckpts_dir, exist_ok=False)

    def print_args(self):
        logging.info(f'Arguments: {self.config}')

    def log(self, data):
        raise NotImplementedError('log method is not implemented.')

    def info(self, message):
        raise NotImplementedError('info method is not implemented.')


class BasicLogger(Logger):
    def __init__(self, config: Config, seed, make_ckpts=True):
        super().__init__(config, seed, make_ckpts=make_ckpts)

        self.log_dir = os.path.join('logs', self.run_group, self.run_name)
        os.makedirs(self.log_dir, exist_ok=False)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(filename)s] => %(message)s',
            handlers=[
                logging.FileHandler(filename=os.path.join(self.log_dir, 'output.log')),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def log(self, data):
        logging.info(f"Class: {data['class']}")
        logging.info(f"CNN Top1: {data['cnn_top1']}")
        logging.info(f"CNN Top5: {data['cnn_top5']}")
        logging.info(f"CNN Average Accuracy: {data['cnn_average_acc']}")

    def info(self, message):
        logging.info(message)


class WandbLogger(Logger):
    def __init__(self, config: Config, seed, project_name='cil-acmap', make_ckpts=True, options=None):
        super().__init__(config, seed, make_ckpts=make_ckpts)

        # Load W&B API Key
        ok = load_dotenv(Path(__file__).resolve().parents[3] / '.env')
        if not ok:
            raise FileNotFoundError('.env could not be found.')

        # Transform config to json
        config_dict = {}
        config_dumped = config.model_dump()
        for key, value in config_dumped.items():
            if isinstance(value, dict):
                config_dict.update(value)
            else:
                config_dict.update({key: value})

        if options is not None:
            config_dict.update(options)

        # Set tags
        tags = [] if not config.debug else ['debug']

        wandb.init(
            project=project_name,
            name=self.run_name,
            group=self.run_group,
            config=config_dict,
            reinit=True,
            tags=tags,
        )

    def log(self, data):
        wandb.log(data)

    def info(self, message):
        print(message)
