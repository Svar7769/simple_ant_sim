# Model settings. One place for every architecture choice.
# Input sizes come from dataset/config.py so the model always matches the data.

from dataclasses import dataclass, asdict, field

from dataset.config import DIR_FEATURES, SELF_FEATURES, OBS_DIM, DIRS, MODES


@dataclass
class ModelConfig:
    arch: str = "transformer"        # "transformer" or "mlp" (baseline)

    # ---- transformer ----
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 1
    use_layernorm: bool = False      # off: residual stream stays linear and easier to read
    dir_embedding: bool = True       # learned embedding per direction (up, down, left, right)
    dir_readout: str = "self"        # "self": direction logits from the self token
                                     # "pointer": one score per direction token
    act: str = "relu"                # MLP activation: "relu" or "gelu"

    # ---- MLP baseline ----
    mlp_hidden: tuple = (128, 128)

    # ---- sizes (from dataset/config.py, do not edit here) ----
    n_dir: int = len(DIRS)                     # 4 direction tokens
    n_dir_features: int = len(DIR_FEATURES)    # 5 features per direction token
    n_self_features: int = len(SELF_FEATURES)  # 3 features in the self token
    obs_dim: int = OBS_DIM                     # 23 flat inputs (MLP baseline)
    n_modes: int = len(MODES)                  # 3: target, trail, wander

    def __post_init__(self):
        assert self.arch in ("transformer", "mlp"), self.arch
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.dir_readout in ("self", "pointer"), self.dir_readout
        assert self.act in ("relu", "gelu"), self.act
        self.mlp_hidden = tuple(self.mlp_hidden)

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def n_tokens(self):
        return self.n_dir + 1            # 4 direction tokens + 1 self token

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


MODEL = ModelConfig()
