import torch
import tiktoken

def save_checkpoint(path, model, optimizer):
    torch.save({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict()}, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])