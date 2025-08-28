from dataclasses import dataclass
import torch


@dataclass
class GptConfig:
    d_model:int = 768
    context_len:int = 1024
    n_layers:int = 12
    vocab_size:int = 50257
    n_heads:int = 12
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    intermidiate_size:int = d_model * 4
    