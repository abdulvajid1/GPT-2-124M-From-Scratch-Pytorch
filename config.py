from dataclasses import dataclass
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

@dataclass
class GptConfig:
    d_model:int = 768
    context_len:int = 64
    n_layers:int = 12
    vocab_size:int = 50257
    n_heads:int = 12
    device: str = device
    intermidiate_size:int = d_model * 4
    n_epoch:int = 5
    lr:float = 1e-4
    batch_size:int = 4
    load_checkpoint = True
    
    