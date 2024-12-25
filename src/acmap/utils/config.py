from pydantic import BaseModel


class AdapterConfig(BaseModel):
    dropout: float
    d_model: int
    down_size: int
    scalar: float
    layernorm_option: str  # none | in | out


class AttentionConfig(BaseModel):
    num_heads: int
    qkv_bias: bool
    attn_drop: float
    proj_drop: float


class BlockConfig(BaseModel):
    dim: int
    dropout: float
    mlp_ratio: float
    ffn_adapt: bool
    ffn_option: str  # parallel | sequential


class TransformerConfig(BaseModel):
    global_pool: bool
    num_classes: int
    embed_dim: int
    out_dim: int
    distilled: bool
    img_size: int
    patch_size: int
    in_chans: int
    depth: int
    drop_rate: float
    drop_path_rate: float
    vpt_on: bool
    vpt_num: int


class OurConfig(BaseModel):
    merge_method: str
    init_random: bool
    init_first_adapter: bool
    trim_rate: float
    use_centroid_map: bool
    limit_centroid_map: int


class ExperimentConfig(BaseModel):
    dataset: str
    shuffle: bool

    name: str
    use_init_ptm: bool
    backbone_type: str

    batch_size: int
    init_lr: float
    later_lr: float
    weight_decay: float
    min_lr: float
    num_workers: int

    scheduler: str
    optimizer: str

    init_epochs: int
    later_epochs: int


class Config(BaseModel):
    block: BlockConfig
    adapter: AdapterConfig
    attention: AttentionConfig
    transformer: TransformerConfig
    our: OurConfig
    exp: ExperimentConfig
    init_cls: int
    increment: int
    seed: int
    device: str
    logger: str  # wandb | basic
    prefix: str
    debug: bool
    ckpts_dir: str
    dataset_dir: str


if __name__ == '__main__':
    import os

    import yaml

    with open(os.path.join('exps', 'cifar.yaml')) as f:
        file = yaml.safe_load(f)

    config = Config.model_validate(file)
    print('Config is valid:', config)

    d = {}
    dd = config.model_dump()
    for k in dd:
        d.update(dd[k])
