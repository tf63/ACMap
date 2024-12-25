import yaml

from acmap.utils import factory
from acmap.utils.config import Config
from acmap.utils.context import Context
from acmap.utils.data_manager import DataManager
from acmap.utils.logger import BasicLogger, WandbLogger
from acmap.utils.options import setup_parser
from acmap.utils.toolkit import count_parameters, set_random


def main():
    args = setup_parser().parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.update(vars(args))

    # Temporary dummy seed value to satisfy validation requirements.
    config.update({'seed': 0})

    config = Config(**config)
    config = Config.model_validate(config)

    for seed in args.seed:
        config.seed = seed
        inference(config=config, seed=seed)


def inference(config: Config, seed):
    # setup device
    set_random(seed)

    # setup datasets
    data_manager = DataManager(
        dataset_name=config.exp.dataset, shuffle=config.exp.shuffle, seed=seed, dataset_dir=config.dataset_dir
    )

    # setup logger
    if config.logger == 'wandb':
        logger = WandbLogger(config, project_name='cil-acmap', seed=seed, make_ckpts=False)
    elif config.logger == 'basic':
        logger = BasicLogger(config, seed, make_ckpts=False)
    else:
        raise ValueError('Invalid logger type.')

    # setup context object
    context = Context(config=config, logger=logger, class_order=data_manager.class_order)

    # setup model
    model = factory.get_model(context=context)
    logger.info(f'All params: {count_parameters(model.network)}')
    logger.print_args()

    cnn_curve = {'top1': [], 'top5': []}
    for task in range(1, context.num_tasks + 1):
        logger.info(f'Task {task}/{context.num_tasks} ========================================================')

        # train (replace cur_adapter with pre-trained adapter)
        model.incremental_eval(data_manager=data_manager)

        # inference
        cnn_accy, _ = model.eval_task()

        # after task
        model.after_task()

        # logging
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve['top5'].append(cnn_accy['top5'])

        data = {
            'task': task,
            'class': task * config.increment,
            'cnn_top1': cnn_accy['top1'],
            'cnn_top5': cnn_accy['top5'],
            'cnn_average_acc': sum(cnn_curve['top1']) / len(cnn_curve['top1']),
        }

        logger.log(data)
        logger.info(f"Top-1 Accuracy: {cnn_accy['top1']}")
        logger.info(f"Average Accuracy: {data['cnn_average_acc']}")


if __name__ == '__main__':
    main()
