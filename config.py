from dataclasses import dataclass
import torch


@dataclass
class GptConfig:
    d_model:int = 768
    context_len:int = 256
    n_layers:int = 12
    vocab_size:int = 55
    n_heads:int = 8
    n_embed:int = 768
    head_dim:int = 96
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    intermidiate_size:int = d_model * 4
    