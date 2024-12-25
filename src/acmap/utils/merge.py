import torch
from torch.nn import ModuleList

from acmap.utils.config import Config


def merge(config: Config, merge_method, prev_adapter: ModuleList, cur_adapter: ModuleList, num_adapters):
    with torch.no_grad():
        prev_state_dict = prev_adapter.state_dict()
        cur_state_dict = cur_adapter.state_dict()

        if merge_method == 'average':
            merged_state_dict = {}
            for key in prev_state_dict.keys():
                if key not in cur_state_dict:
                    raise ValueError(f'Warning {key} not in cur_adapter')

                merged_state_dict[key] = (prev_state_dict[key] * num_adapters + cur_state_dict[key]) / (
                    num_adapters + 1
                )

            return merged_state_dict
        elif merge_method == 'trim':
            merged_state_dict = {}
            for key in prev_state_dict.keys():
                if key not in cur_state_dict:
                    raise ValueError(f'Warning {key} not in cur_adapter')

                all_params = torch.cat([torch.abs(p.view(-1)) for p in cur_state_dict[key]])
                sorted_params, _ = torch.sort(all_params)

                threshold_index = min(int(config.our.trim_rate * len(sorted_params)), len(sorted_params) - 1)
                threshold = sorted_params[threshold_index]

                mask = torch.ones_like(cur_state_dict[key])
                mask[torch.abs(cur_state_dict[key]) < threshold] = 0

                scale = torch.ones_like(cur_state_dict[key]) * (1.0 / (num_adapters + 1))
                scale[mask < 0.1] = 1.0 / num_adapters

                merged_state_dict[key] = (prev_state_dict[key] * num_adapters + cur_state_dict[key] * mask) * scale

            return merged_state_dict
        elif merge_method == 'aper':
            return prev_state_dict
        elif merge_method == 'none':
            return cur_state_dict
        else:
            raise ValueError(f'{merge_method} is not implemented')


def initial_merge(config: Config, merge_method, cur_adapter: ModuleList):
    with torch.no_grad():
        cur_state_dict = cur_adapter.state_dict()
        if merge_method == 'average':
            return cur_state_dict
        elif merge_method == 'trim':
            merged_state_dict = {}

            all_params = torch.cat([torch.abs(p.view(-1)) for p in cur_adapter.parameters()])
            sorted_params, _ = torch.sort(all_params)

            threshold_index = min(int(config.our.trim_rate * len(sorted_params)), len(sorted_params) - 1)
            threshold = sorted_params[threshold_index]

            for key in cur_state_dict.keys():
                if key not in cur_state_dict:
                    raise ValueError(f'Warning {key} not in cur_adapter')

                mask = torch.ones_like(cur_state_dict[key])
                mask[torch.abs(cur_state_dict[key]) < threshold] = 0

                merged_state_dict[key] = cur_state_dict[key] * mask

            return merged_state_dict
        elif merge_method == 'aper':
            return cur_state_dict
        elif merge_method == 'none':
            return cur_state_dict
        else:
            raise ValueError(f'{merge_method} is not implemented')
