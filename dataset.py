from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

class GPTDataset(Dataset):
    def __init__(self, path='input.txt', tokenizer=None):
        super().__init__()
        with open(path, 'r') as f:
            data = f.read()
        
        data = torch.tensor(tokenizer.encode(data))
            

        
        