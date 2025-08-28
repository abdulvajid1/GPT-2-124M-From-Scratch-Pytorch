import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GptConfig
import math 
from huggingface_hub import PyTorchModelHubMixin
import tiktoken


class PositionalEncoding(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(config.context_len, config.d_model)
    
    def forward(self, x):
        seq_len = x.shape[1] 
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)
  
  
class FeedForward(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.intermidiate_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.intermidiate_size, config.d_model)    
        )
        
    def forward(self, x):
        return self.feed_forward(x)
 
    
class RMSNorm(nn.Module):
    def __init__(self, config: GptConfig, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.d_model))
        self.eps = eps
    def forward(self, x: torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return x_norm * self.scale



class GPTAttention(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.config = config
        self.Q_w = nn.Linear(config.d_model, config.d_model)
        self.K_w = nn.Linear(config.d_model, config.d_model)
        self.V_w = nn.Linear(config.d_model, config.d_model)
        self.O_w = nn.Linear(config.d_model, config.d_model)
        causal_mask = torch.tril(torch.ones(1, 1, config.context_len, config.context_len))
        self.register_buffer('mask', causal_mask)
        
    def _mask_atten_scores(self, atten_score, seq_len):
        # Masking (Casual)
        atten_score = atten_score.masked_fill(self.mask[:, :, :seq_len, :seq_len]==0, float('-inf'))
        return atten_score
    
    def _calc_atten_score(self, query, key):
        atten_score = (query @ key.transpose(-1, -2)) *  (1.0/ math.sqrt(key.size(-1)))
        return atten_score
    
    def _create_contextualized_embeds(self, atten_score, value):
        contextulized_embeds = atten_score @ value # (b, n_head, seq, seq) * (b, n_head, seq, head_dim) -> (b, n_head, seq, head_dim)
        contextulized_embeds = contextulized_embeds.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.config.n_heads * self.head_dim)
        return contextulized_embeds
        
    def forward(self, x: torch.Tensor):
        self.batch_size, self.seq_len = x.shape[0], x.shape[1]
        
        assert self.config.d_model % self.config.n_heads == 0, "d_model should divisible by n_heads"
        
        self.head_dim = self.config.d_model // self.config.n_heads
        
        query = self.Q_w(x)
        key = self.K_w(x)
        value = self.V_w(x)
        
        # (b, s, d_model) -> view -> (batch_size, seq_len, num_head, head_dim) -> permute ->(b, num_head, s, head_dim)
        query = query.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(self.batch_size, self.seq_len, self.config.n_heads, self.head_dim).transpose(1, 2)
        
        atten_score = self._calc_atten_score(query, key)
        atten_score = self._mask_atten_scores(atten_score, self.seq_len)
        atten_score = F.softmax(atten_score, dim=-1)
        contextualized_embed = self._create_contextualized_embeds(atten_score, value)
        return self.O_w(contextualized_embed)


class DecoderBlock(nn.Module):
    def __init__(self, config: GptConfig):
        super().__init__()
        
        self.attention = GPTAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = RMSNorm(config)
        self.norm2 = RMSNorm(config)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
    

class GPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: GptConfig):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = PositionalEncoding(config)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_rms_norm = RMSNorm(config)
        self.final_layer = nn.Linear(config.d_model, config.vocab_size)
        self.final_layer.weight = self.tok_embed.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    
    def forward(self, x, target=None):
        """x: (b, seq)
           target: (b, seq)"""
        batch_size, seq_len = x.size()
        assert seq_len <= self.config.context_len, "seq length should be less than context length"
        x = self.tok_embed(x)
        x = self.pos_embed(x)
        
        for block in self.decoder_blocks:
            x = block(x)
            
        x = self.final_rms_norm(x)
        logits = self.final_layer(x) # (b, seq, vocab_size)
        loss = None
        if target != None:
            loss = F.cross_entropy(logits.view(batch_size*seq_len, -1), target.view(-1))
        
        return {'logits' : logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, text, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        idx = torch.tensor(self.tokenizer.encode_batch(text)).to(self.config.device)
        max_len = max_new_tokens if max_new_tokens else self.config.context_len
        
        for _ in range(max_len):
            logits = self(idx, target=None)['logits']
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)  
            if top_k is not None:
                topk_probs, topk_vocab_indieces = torch.topk(probs, top_k, dim=-1)
                next_token_idx = torch.multinomial(topk_probs, num_samples=1)
                next_token = torch.gather(topk_vocab_indieces, -1, next_token_idx)
                idx = torch.cat([idx, next_token], dim=-1)
            else: 
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=-1)
        idx = idx.detach().cpu().tolist()
        out = self.tokenizer.decode_batch(idx)
        return out
    
    
    
def main():
    config = GptConfig()
    gpt = GPT(config)
    gpt.eval()
    gpt.to(config.device)
    sample_input = ['hai what is'] * 5
    out_list = gpt.generate(sample_input, max_new_tokens=15)
    for out in out_list:
        print(out)
    

if __name__ == '__main__':
    main()

