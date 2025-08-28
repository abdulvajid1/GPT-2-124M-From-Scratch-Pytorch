from dataclasses import dataclass
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f'Using Device: {device}')

@dataclass
class GptConfig:
    d_model:int = 768
    context_len:int = 1024
    n_layers:int = 12
    vocab_size:int = 50257
    n_heads:int = 12
    device: str = device
    intermidiate_size:int = d_model * 4
    