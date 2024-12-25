import copy
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from acmap.backbone.linears import CosineLinear
from acmap.backbone.vit_acmap import vit_base_patch16_224, vit_base_patch16_224_in21k
from acmap.utils.context import Context
from acmap.utils.merge import initial_merge, merge


class ACMapNet(nn.Module):
    def __init__(self, context: Context):
        super().__init__()
        self.c = context
        self.config = self.c.config
        self.logger = self.c.logger
        self.out_dim = self.config.transformer.out_dim
        self.use_init_ptm = self.config.exp.use_init_ptm
        self.device = self.config.device

        self.backbone = self.get_backbone()

        self.fc: Optional[CosineLinear] = None
        self.merged_adapter_list: List[Optional[nn.ModuleList]] = []
        self.protos_list_source = []

    def get_backbone(self):
        backbone_type = self.config.exp.backbone_type.lower()

        self.logger.info('Loading the pretrained model from timm...')
        if backbone_type == 'vit_base_patch16_224':
            model = vit_base_patch16_224(config=self.config)
        elif backbone_type == 'vit_base_patch16_224_in21k':
            model = vit_base_patch16_224_in21k(config=self.config)
        else:
            raise NotImplementedError(f'Unknown type {backbone_type}')

        return model

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def copy(self):
        return copy.deepcopy(self)

    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * 2
        else:
            return self.out_dim * 1

    def update_proxy_fc(self):
        self.proxy_fc = self.generate_fc(self.out_dim, self.c.cur_task_size).to(self.device)

    # (proxy_fc = cls * dim)
    def update_fc(self):
        new_fc = self.generate_fc(self.feature_dim, self.c.total_classes).to(self.device)
        new_fc.reset_parameters_to_zero()

        if self.fc is None:
            self.fc = new_fc
            return

        old_nb_classes = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        new_fc.sigma.data = self.fc.sigma.data
        if new_fc.weight.shape[1] != weight.shape[1]:
            new_fc.weight.data[:old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        else:
            new_fc.weight.data[:old_nb_classes, :] = nn.Parameter(weight)

        self.fc = new_fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def replace_fc(self, dataset, data_loader):
        assert self.fc is not None

        ptm_index = 0
        protos_current = self.extract_prototype(dataset, data_loader, self.merged_adapter_list[-1])

        if (
            self.config.our.use_centroid_map
            and len(self.merged_adapter_list) > 1
            and len(self.merged_adapter_list) < self.config.our.limit_centroid_map
        ):
            self.centroid_mapping(dataset, data_loader, protos_current, ptm_index)

        self.fc.weight.data[-protos_current.shape[0] :, self.out_dim * ptm_index : self.out_dim * (ptm_index + 1)] = (
            protos_current
        )

        self.protos_list_source.append(protos_current)

        if self.use_init_ptm:
            ptm_index = 1
            protos = self.extract_prototype(dataset, data_loader, adapter=None)

            self.fc.weight.data[-protos.shape[0] :, self.out_dim * ptm_index : self.out_dim * (ptm_index + 1)] = protos

    def centroid_mapping(self, dataset, data_loader, protos_current, ptm_index):
        assert self.fc is not None

        protos_list = []
        for adapter in self.merged_adapter_list[:-1]:
            protos = self.extract_prototype(dataset, data_loader, adapter)
            protos_list.append(protos)

        protos_list_mapped = []
        for i, protos in enumerate(self.protos_list_source):
            proto_shift = torch.mean(protos_current, dim=0).to(self.device) - torch.mean(protos_list[i], dim=0).to(
                self.device
            )

            protos_mapped = protos.to(self.device) + proto_shift.to(self.device)
            protos_list_mapped.append(protos_mapped)

        fc_mapped = torch.cat(protos_list_mapped, dim=0)

        self.fc.weight.data[: -protos_current.shape[0], self.out_dim * ptm_index : self.out_dim * (ptm_index + 1)] = (
            fc_mapped
        )

    def extract_prototype(self, dataset, data_loader, adapter):
        with torch.no_grad():
            prog_bar = tqdm(data_loader)

            # extract embeddings
            embedding_list, label_list = [], []
            for _, batch in enumerate(prog_bar):
                (_, data, label) = batch
                data = data.to(self.device)
                label = label.to(self.device)
                embedding = self.backbone.forward_proto(data, adapter)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                prog_bar.set_description('Extracting prototype from PTM =>')

            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            # construct prototype-based classifier
            class_list = np.unique(dataset.labels)
            proto_list = []
            for class_index in class_list:
                # calculate prototype
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                embedding = embedding_list[data_index]
                proto = embedding.mean(0)
                proto_list.append(proto[None])

            return torch.cat(proto_list, dim=0)

    def calc_cosine_similarity(self, dataset, data_loader, adapter_source, adapter_target):
        proto_source = self.extract_prototype(dataset, data_loader, adapter_source).flatten()
        proto_target = self.extract_prototype(dataset, data_loader, adapter_target).flatten()

        cosine_similarity = F.cosine_similarity(proto_source, proto_target, dim=0).item()

        return cosine_similarity

    def merge_adapters(self):
        if len(self.merged_adapter_list) >= self.config.our.limit_centroid_map:
            self.logger.info('Skip merge')
            return

        merge_method = self.config.our.merge_method
        self.logger.info(f'Merge method: {merge_method}')

        # not merge
        if merge_method == 'simple':
            self.merged_adapter_list.append(None)
            return

        # merge adapters
        if self.c.is_first_task:
            # initialize merged adpater
            merged_state_dict = initial_merge(
                config=self.config,
                merge_method=merge_method,
                cur_adapter=self.backbone.cur_adapter,
            )
        else:
            # obtain merged adpater
            assert self.merged_adapter_list[-1] is not None
            merged_state_dict = merge(
                config=self.config,
                merge_method=merge_method,
                prev_adapter=self.merged_adapter_list[-1],
                cur_adapter=self.backbone.cur_adapter,
                num_adapters=len(self.merged_adapter_list),
            )

        with torch.no_grad():
            merged_adapter = copy.deepcopy(self.backbone.cur_adapter).requires_grad_(False)
            merged_adapter.load_state_dict(merged_state_dict)
            self.merged_adapter_list.append(merged_adapter)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if not test:
            # Training
            x = self.backbone.forward_train(x)
            out = self.proxy_fc(x)
        else:
            # Testing
            assert self.fc is not None
            adapter_list = [self.merged_adapter_list[-1]]
            if self.use_init_ptm:
                adapter_list = adapter_list + [None]

            x = self.backbone.forward_test(x, adapter_list)
            out = self.fc(x)

        out.update({'features': x})
        return out
