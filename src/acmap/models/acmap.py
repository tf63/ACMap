import os

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from acmap.utils.context import Context
from acmap.utils.inc_net import ACMapNet
from acmap.utils.toolkit import accuracy, tensor2numpy

TOP_K = 5


class Learner:
    def __init__(self, context: Context):
        self.c = context
        self.config = self.c.config
        self.logger = self.c.logger
        self.network = ACMapNet(context=self.c)

        self.num_workers = self.config.exp.num_workers
        self.batch_size = self.config.exp.batch_size
        self.device = self.config.device
        self.init_lr = self.config.exp.init_lr
        self.later_lr = self.config.exp.later_lr
        self.min_lr = self.config.exp.min_lr
        self.weight_decay = self.config.exp.weight_decay
        self.init_epochs = self.config.exp.init_epochs
        self.later_epochs = self.config.exp.later_epochs
        self.init_first_adapter = self.config.our.init_first_adapter
        self.use_init_ptm = self.config.exp.use_init_ptm
        self.optimizer_name = self.config.exp.optimizer
        self.scheduler_name = self.config.exp.scheduler

    def after_task(self):
        self.network.freeze()

        if self.c.is_first_task and self.init_first_adapter:
            self.network.backbone.init_adapter = self.network.backbone.cur_adapter

        self.c.next_task()

    def incremental_train(self, data_manager):
        # setup train
        self.network.update_proxy_fc()
        self.network.backbone.setup_adapter()

        # setup data
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self.c.known_classes, self.c.total_classes), source='train', mode='train'
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.test_dataset = data_manager.get_dataset(np.arange(0, self.c.total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self.c.known_classes, self.c.total_classes), source='train', mode='test'
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        self._train(self.train_loader)

        # merge adapter
        self.network.merge_adapters()

        # replace fc
        self.network.update_fc()
        self.network.replace_fc(dataset=self.train_dataset_for_protonet, data_loader=self.train_loader_for_protonet)

    def _train(self, train_loader):
        self.network.to(self.device)
        if self.c.is_first_task:
            optimizer = self.get_optimizer(lr=self.init_lr)
            scheduler = self.get_scheduler(optimizer, self.init_epochs)
        else:
            optimizer = self.get_optimizer(lr=self.later_lr)
            scheduler = self.get_scheduler(optimizer, self.later_epochs)

        self._init_train(train_loader, optimizer, scheduler)

    def incremental_eval(self, data_manager):
        # setup train
        self.network.update_proxy_fc()
        self.network.backbone.setup_adapter()

        # setup data
        self.data_manager = data_manager
        self.test_dataset = data_manager.get_dataset(np.arange(0, self.c.total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self.c.known_classes, self.c.total_classes), source='train', mode='test'
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        # load adapter
        self.network.to(self.device)
        ckpts_path = os.path.join(
            self.config.ckpts_dir,
            self.logger.run_group,
            f'acmap-average-in21k-seed{self.config.seed}',
            f'task{self.c.cur_task}.pkl',
        )
        ckpt_model = torch.load(ckpts_path)
        self.network.backbone.cur_adapter.load_state_dict(ckpt_model['state_dict'])

        # merge adapter
        self.network.merge_adapters()

        # replace fc
        self.network.update_fc()
        self.network.replace_fc(dataset=self.train_dataset_for_protonet, data_loader=self.train_loader_for_protonet)

    def get_optimizer(self, lr):
        if self.optimizer_name == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.parameters()), lr=lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.network.parameters()), lr=lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Invalid optimizer name: {self.optimizer_name}')

        return optimizer

    def save_checkpoint(self, ckpt_dir):
        save_dict = {
            'tasks': self.c.cur_task,
            'state_dict': self.network.backbone.cur_adapter.state_dict(),
        }

        ckpt_path = os.path.join(ckpt_dir, f'task{self.c.cur_task}.pkl')
        self.logger.info(f'saving checkpoint to {ckpt_path}')
        torch.save(save_dict, ckpt_path)

        if self.c.is_first_task:
            if self.init_first_adapter:
                save_dict = {'state_dict': self.network.backbone.cur_adapter.state_dict()}
            else:
                save_dict = {'state_dict': self.network.backbone.init_adapter.state_dict()}

            ckpt_path = os.path.join(ckpt_dir, 'init.pkl')
            self.logger.info(f'saving checkpoint to {ckpt_path}')
            torch.save(save_dict, ckpt_path)

    def get_scheduler(self, optimizer, epoch):
        if self.scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.scheduler_name == 'constant':
            scheduler = None
        else:
            raise ValueError(f'Invalid scheduler name: {self.scheduler_name}')

        return scheduler

    def _init_train(self, train_loader, optimizer, scheduler):
        if self.c.is_first_task:
            epochs = self.init_epochs
        else:
            epochs = self.later_epochs

        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self.network.train()

            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self.c.known_classes >= 0,
                    aux_targets - self.c.known_classes,
                    -1,
                )

                output = self.network(inputs, test=False)
                logits = output['logits']

                loss = F.cross_entropy(logits, aux_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = f'Task {self.c.cur_task}, Epoch {epoch+1}/{epochs} => Loss {losses / len(train_loader):.3f}, Train_accy {train_acc:.2f}'
            prog_bar.set_description(info)

        self.logger.info(info)

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)['logits']
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self.c.known_classes, self.c.cur_task_size)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret[f'top{TOP_K}'] = np.around(
            (np.tile(y_true, (TOP_K, 1)) == y_pred.T).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)

        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        calc_task_acc = True

        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0

        prog_bar = tqdm(loader)

        self.network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.network.forward(inputs, test=True)['logits']
            predicts = torch.topk(outputs, k=TOP_K, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            # calculate the accuracy by using task_id
            if calc_task_acc:
                init_cls = self.c.increments[0]
                increment = self.c.cur_task_size
                task_ids = (targets - init_cls) // increment + 1
                task_logits = torch.zeros(outputs.shape).to(self.device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = init_cls
                    else:
                        start_cls = init_cls + (task_id - 1) * increment
                        end_cls = init_cls + task_id * increment
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - init_cls) // increment + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()

                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

            prog_bar.set_description('Eval =>')

        if calc_task_acc:
            self.logger.info(f'Task correct: {tensor2numpy(task_correct) * 100 / total}')
            self.logger.info(f'Task acc: {tensor2numpy(task_acc) * 100 / total}')

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
