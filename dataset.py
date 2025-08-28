from torch.utils.data import Dataset, DataLoader
import torch

class GPTDataset(Dataset):
    def __init__(self, path='input.txt', context_len=1024, tokenizer=None):
        super().__init__()
        with open(path, 'r') as f:
            data = f.read()
        
        context_len+=1
        data = tokenizer.encode(data)
        self.data = torch.tensor([data[i: i+context_len] for i in range(0, len(data) - context_len, context_len)])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        index_data = self.data[index]
        x = index_data[:-1]
        y = index_data[1:]
        return x, y

def get_data_loader(tokenizer,context_len=1024, batch_size=8, shuffle=True, pin_memory=False,num_workers=1):
    dataset = GPTDataset(context_len=context_len, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    return data_loader
    
    


def main():
    import tiktoken
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(tokenizer=tokenizer, context_len=5)
    print(dataset[5])
    print(f"{len(dataset)}")
    loader = get_data_loader(tokenizer)
    for x, y in loader:
        print(x, y)
        break
    
    
if __name__ == "__main__":
    main()
    

        
        