# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
import copy
import math
from functools import partial
from typing import List, Optional

# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import timm
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed

from acmap.utils.config import Config


class Adapter(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_embd = config.adapter.d_model
        self.down_size = config.adapter.down_size
        self.layernorm_option = config.adapter.layernorm_option
        self.scale = config.adapter.scalar

        if self.layernorm_option == 'in' or self.layernorm_option == 'out':
            self.layer_norm_before = nn.LayerNorm(self.n_embd)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = config.adapter.dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual

        if self.layernorm_option == 'in':
            x = self.layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        up = self.up_proj(down)
        up = up * self.scale

        if self.layernorm_option == 'out':
            up = self.layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dim = config.block.dim
        self.num_heads = config.attention.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.dim, self.dim, bias=config.attention.qkv_bias)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=config.attention.qkv_bias)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=config.attention.qkv_bias)

        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(config.attention.proj_drop)
        self.attn_drop = nn.Dropout(config.attention.attn_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = self.attn_drop(attn_weights)

        attn_output = torch.bmm(attn_probs, v)
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, config: Config, _layer_id):
        super().__init__()
        self.config = config
        self.dim = config.block.dim

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)

        self.attn = Attention(config=config)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop = config.block.dropout

        dpr = [x.item() for x in torch.linspace(0, config.transformer.drop_path_rate, config.transformer.depth)]
        drop_path = dpr[_layer_id]
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp_ratio = config.block.mlp_ratio
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.fc1 = nn.Linear(self.dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, self.dim)
        self.act = nn.GELU()
        self.mlp_drop = nn.Dropout(self.drop)

    def forward(self, x, adapt=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x

        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt is not None and self.config.block.ffn_adapt:
            if self.config.block.ffn_option == 'sequential':
                x = adapt(x)
            elif self.config.block.ffn_option == 'parallel':
                x = x + adapt(residual, add_residual=False)
            else:
                raise ValueError(self.config.block.ffn_option)

        x = residual + x

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.depth = config.transformer.depth
        self.device = config.device

        self.num_classes = config.transformer.num_classes
        self.num_features = self.embed_dim = (
            config.transformer.embed_dim
        )  # num_features for consistency with other models
        self.num_tokens = 2 if config.transformer.distilled else 1

        self.distilled = config.transformer.distilled
        self.global_pool = config.transformer.global_pool

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(self.embed_dim)

        self.patch_embed = PatchEmbed(
            img_size=config.transformer.img_size,
            patch_size=config.transformer.patch_size,
            in_chans=config.transformer.in_chans,
            embed_dim=self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if self.distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
        self.pos_drop = nn.Dropout(p=config.transformer.drop_rate)

        # stochastic depth decay rule
        self.blocks = nn.Sequential(*[Block(config=config, _layer_id=i) for i in range(self.depth)])

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None
        if self.distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # MAE begins
        if self.global_pool:
            self.fc_norm = norm_layer(self.embed_dim)

            del self.norm  # remove the original norm

        # Adapter begins
        if config.transformer.vpt_on:
            assert config.transformer.vpt_num > 0, config.transformer.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, config.transformer.vpt_num, self.embed_dim)) for _ in range(self.depth)]
            )
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        # setup adapter
        self.init_adapter = self.construct_adapter().requires_grad_(False)
        self.setup_adapter()

    def init_weights(self, mode=''):
        raise NotImplementedError()

    @torch.jit.ignore  # type: ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad_(True)

    def construct_adapter(self):
        adapter = nn.ModuleList()

        if self.config.block.ffn_adapt:
            for _ in range(len(self.blocks)):
                _adapter = Adapter(self.config).to(self.device)
                adapter.append(_adapter)

        return adapter

    def setup_adapter(self):
        if self.config.our.init_random:
            self.cur_adapter = self.construct_adapter().requires_grad_(True)
        else:
            self.cur_adapter = copy.deepcopy(self.init_adapter).requires_grad_(True)

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if self.config.transformer.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x, self.cur_adapter[idx])
            if self.config.transformer.vpt_on:
                x = x[:, self.config.transformer.vpt_num :, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_test(self, x, adapter_list: List[Optional[nn.ModuleList]]):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        features = []

        for adapter in adapter_list:
            x = copy.deepcopy(x_init)
            for i in range(len(self.blocks)):
                if adapter is None:
                    adapt = None
                else:
                    adapt = adapter[i]
                x = self.blocks[i](x, adapt)
            x = self.norm(x)
            features.append(x)

        output = torch.Tensor().to(features[0].device)
        for x in features:
            cls = x[:, 0, :]
            output = torch.cat((output, cls), dim=1)

        return output

    def forward_proto(self, x, adapter: Optional[nn.ModuleList] = None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # the init_PTM's feature
        if adapter is None:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)
            x = self.norm(x)
            output = x[:, 0, :]
            return output

        x = copy.deepcopy(x_init)
        for i in range(len(self.blocks)):
            adapt = adapter[i]
            x = self.blocks[i](x, adapt)
        x = self.norm(x)
        output = x[:, 0, :]

        return output


def vit_base_patch16_224(config: Config):
    model = VisionTransformer(config=config)

    checkpoint_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768 : 768 * 2]
            v_weight = qkv_weight[768 * 2 :]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768 : 768 * 2]
            v_bias = qkv_bias[768 * 2 :]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k(config: Config):
    model = VisionTransformer(config=config)

    checkpoint_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # modify the checkpoint state dict to match the model
    # first, split qkv weight into q, k, v
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768 : 768 * 2]
            v_weight = qkv_weight[768 * 2 :]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768 : 768 * 2]
            v_bias = qkv_bias[768 * 2 :]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    # second, modify the mlp.fc.weight to match fc.weight
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)

    # freeze all but the adapter
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model
