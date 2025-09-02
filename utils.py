import torch
import tiktoken

def save_checkpoint(path, model, optimizer, global_step):
    torch.save({'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'global_step': global_step}, path)

def load_checkpoint(model, optimizer=None, path='checkpoints/ckpt.pt',):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer!=None:
        optimizer.load_state_dict(checkpoint['optim_state'])

    global_step = checkpoint.get('global_step', 0)
    
    return global_step 