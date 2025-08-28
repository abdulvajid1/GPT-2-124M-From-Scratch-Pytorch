import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GptConfig
import math 


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)
    
    def forward(self, x):
        seq_len = x.shape[1] 
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)
  
  
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.intermidiate_size),
            nn.GELU(),
            nn.Linear(config.intermidiate_size, config.d_model)    
        )
        
    def forward(self, x):
        return self.feed_forward(x)
 
    
class RMSNorm(nn.Module):
    def __init__(self, config, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.d_model))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return x_norm * self.scale



class GPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.Q_w = nn.Linear(config.d_model, config.d_model)
        self.K_w = nn.Linear(config.d_model, config.d_model)
        self.V_w = nn.Linear(config.d_model, config.d_model)
        self.O_w = nn.Linear(config.d_model, config.d_model)
        
    def _mask_atten_scores(self, atten_score, seq_len):
        # Masking (Casual)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len)).to(self.config.device)  # lower triangle
        atten_score = atten_score.masked_fill(mask==0, float('-inf'))
        return atten_score
    
    def _calc_atten_score(self, query, key):
        d_k = key.size(-1)
        atten_score = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        return atten_score
    
    def _create_contextualized_embeds(self, atten_score, value):
        contextulized_embeds = atten_score @ value # (b, n_head, seq, seq) * (b, n_head, seq, d_model) -> (b, n_head, seq, d_model)
        contextulized_embeds = contextulized_embeds.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.config.n_heads * self.config.head_dim)
        return contextulized_embeds
        
    def forward(self, x: torch.Tensor):
        self.batch_size, self.seq_len = x.shape[0], x.shape[1]
        
        query = self.Q_w(x)
        key = self.K_w(x)
        value = self.V_w(x)
        
        # (b, s, d_model) -> view -> (batch_size, seq_len, num_head, head_dim) -> permute ->(b, num_head, s, head_dim)
        query = query.view(self.batch_size, self.seq_len, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        key = key.view(self.batch_size, self.seq_len, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        value = value.view(self.batch_size, self.seq_len, self.config.n_heads, self.config.head_dim).transpose(1, 2)
        
        atten_score = self._calc_atten_score(query, key)
        atten_score = self._mask_atten_scores(atten_score, self.seq_len)
        atten_score = F.softmax(atten_score, dim=-1)
        contextualized_embed = self._create_contextualized_embeds(atten_score, value)
        return self.O_w(contextualized_embed)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = GPTAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        
    def forward(self, x):
        original_x = x
        x = self.norm1(x)
        x = self.attention(x) + original_x
        
        original_x = x
        x = self.norm2(x)
        x = self.feed_forward(x) + original_x
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.tok_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = PositionalEncoding(config)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_rms_norm = RMSNorm(config)
        self.final_layer = nn.Linear(config.n_embed, config.vocab_size)
        self.criteria = nn.CrossEntropyLoss()
    
    def forward(self, x, target):
        """x: (b, seq)
           target: (b, seq)"""
        batch_size, seq_len = x.size()
        x = self.tok_embed(x)
        x = self.pos_embed(x)
        
        for block in self.decoder_blocks:
            x = block(x)
            
        x = self.final_rms_norm(x)
        logits = self.final_layer(x) # (b, seq, vocab_size)
        loss = self.criteria(logits.view(batch_size*seq_len, -1), target.view(-1))
        return {'logits' : logits, "loss": loss}
    
    

def main():
    config = GptConfig()
    sample = torch.randint(1, 10, size=(1, 5)).to('cuda')
    target = torch.randint(1, 10, size=(1, 5)).to('cuda')
    gpt = GPT(config).to('cuda')
    print('logits shape', gpt(sample, target)['logits'].shape)
    print('loss', gpt(sample, target)['loss'])
    

if __name__ == '__main__':
    main()

