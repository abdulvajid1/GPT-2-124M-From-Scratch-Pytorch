import torch
import tiktoken
import os
from model import GPT


def save_optimizer(optimizer, global_step, path):
    path = os.path.join(path, 'optimizer.pt')    
    torch.save({'optim_state': optimizer.state_dict(), 'global_step': global_step}, path)


def load_checkpoint(config, checkpoint_path, device='cpu'):
    
    # Load model from full path
    model = GPT.from_pretrained(checkpoint_path, config=config).to(device)
    optimizer = model.configure_optimizer(config.lr, weight_decay=config.weight_decay)
    
    # Load optimizer from base path
    save_path, _ = os.path.split(checkpoint_path)
    optimizer_path = os.path.join(save_path, 'optimizer.pt')
    
    optimizer_checkpoint = torch.load(optimizer_path)
    optimizer.load_state_dict(optimizer_checkpoint['optim_state'])
    
    # load global step if exist
    global_step = optimizer_checkpoint.get('global_step', 0)
    return model, optimizer, global_step 


if __name__ == '__main__':
    SAVE_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints')
    import code; code.interact(local=locals())