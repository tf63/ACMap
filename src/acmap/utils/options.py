import argparse
import os


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms')
    parser.add_argument(
        '--config', type=str, default=os.path.join('exps', 'cifar.yaml'), help='Path to the yaml file of settings'
    )
    parser.add_argument('--init_cls', type=int, default='0', help='Number of classes at the start of training')
    parser.add_argument('--increment', type=int, default='5', help='Number of classes to increment per task')
    parser.add_argument('--seed', type=int, nargs='+', default=[1993], help='List of seed values')
    parser.add_argument('--device', type=str, default='cuda', help='Device type (e.g. cpu, cuda, mps)')
    parser.add_argument('--logger', type=str, default='basic', help='Logger type (basic|wandb)')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for the exp name')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument(
        '--dataset_dir', type=str, default='dataset', help='Directory path where the dataset is stored'
    )
    parser.add_argument(
        '--ckpts_dir',
        type=str,
        default=os.path.join('data', 'acmap', 'ckpts'),
        help='Directory path where checkpoints will be saved',
    )

    return parser
