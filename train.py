from config import GptConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import get_data_loader
from model import GPT
import tiktoken
from utils import load_checkpoint, save_checkpoint
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer,config, loader, epoch, writer, save_step, save_path):
    progress_bar = tqdm(enumerate(loader), leave=True, desc=f'Epoch {epoch}: ')
    for step, (x, y) in progress_bar:
        global_step = epoch*len(loader) + step
        x, y = x.to(config.device), y.to(config.device)
        
        optimizer.zero_grad()
        loss = model(x, y)['loss']
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
        
        writer.add_scalar('Training loss', loss.item(), global_step)
        
        if (step+1) % save_step == 0:
            save_checkpoint(path=save_path,
                model=model,
                optimizer=optimizer)
        
        

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
         device = 'mps'

    print(f'Using Device: {device}')
    
    config = GptConfig()
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPT(config).to(config.device)
    loader = get_data_loader(tokenizer, config.context_len, config.batch_size, num_workers=4, pin_memory=True)
    print(f'Total number of samples: {len(loader)}')
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(.9, 0.999))
    save_path = 'checkpoints'
    save_step = 500
    save_path = 'checkpoints/ckpt.pt'
    writer = SummaryWriter(log_dir='runs')
    
    if config.load_checkpoint:
        load_checkpoint(path=save_path, model=model, optimizer=optimizer)
    
    for epoch in range(config.n_epoch):
        train(model, optimizer,config, loader, epoch, writer, save_step, save_path)
        
        # generate sample after each epoch
        sample_input = ['you are a'] * 5
        out_list = model.generate(sample_input, temperature=1.0)
        for out in out_list:
            print(out)

if __name__ == "__main__":
    main()
            