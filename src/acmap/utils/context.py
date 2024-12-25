from typing import List

from acmap.utils.config import Config
from acmap.utils.logger import Logger


class Context:
    def __init__(self, config: Config, logger: Logger, class_order: List[int]):
        self.config = config
        self.logger = logger

        self.class_order = class_order

        # setup dataset split
        self.increments = [config.init_cls] if config.init_cls > 0 else [config.increment]
        assert self.increments[0] <= len(class_order), 'No enough classes.'
        while sum(self.increments) + config.increment < len(self.class_order):
            self.increments.append(config.increment)
        offset = len(self.class_order) - sum(self.increments)
        if offset > 0:
            self.increments.append(offset)

        # setup cil
        self.cur_task = 1
        self.known_classes = 0

    def next_task(self):
        if self.cur_task < self.num_tasks - 1:
            self.cur_task += 1
            self.known_classes += self.cur_task_size

    @property
    def total_classes(self):
        return self.known_classes + self.cur_task_size

    @property
    def num_classes(self):
        return len(self.class_order)

    @property
    def num_tasks(self):
        return len(self.increments)

    @property
    def cur_task_size(self):
        return self.increments[self.cur_task - 1]

    @property
    def is_first_task(self):
        return self.cur_task == 1
