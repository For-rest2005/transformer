import torch 
import torch.nn as nn
from torch.nn import functional as F
import random
from transformers import AutoTokenizer

#hparam
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 1024
batch_size = 2
n_embd = 256
learning_rate = 0.03
max_iters = 10
eval_interval = int(max_iters/10)
eval_iters = 20
dropout_value = 0.1
num_heads = 16
head_size = n_embd // num_heads
n_layer = 16
max_new_tokens = 512

torch.manual_seed(1337)
file_name = "dataset.txt"
tokenizer_path = "./tokenizer"

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# stoi = {ch:i for i,ch in enumerate(chars)}
# itos = {i:ch for i,ch in enumerate(chars)}
# encode = lambda str1: [stoi[c] for c in str1] #string to number string
# decode = lambda list1: "".join([itos[i] for i in list1])

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("Tokenizer load successfully")

vocab_size = tokenizer.vocab_size
encode = lambda str1: tokenizer(str1)['input_ids']
decode = lambda list1: tokenizer.decode(list1)

data = torch.tensor(encoder(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"file {file_name} load success")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)

    return x,y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  
        # why mask? you can't rely on the next token to predict the next token
        # sometimes not. such as translation and summerization
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = q @ k.transpose(-2, -1) * (k.shape[-1]**(-0.5))
        weight = weight.masked_fill(self.tril == 0, float("-inf"))
        #avoid the grad explosion
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight) 
        
        v = self.value(x)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4), nn.ReLU(), nn.Linear(n_embd*4, n_embd), nn.Dropout(dropout_value)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#------
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # initialized with random values
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, target=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T, device=device)
        position_embd = self.position_embedding_table(position_idx)

        x = token_embd + position_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # flatten
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens): # token_sequ: already input
        for _ in range(max_new_tokens):
            token_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(token_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1) # depend on the probability to sample
            token_sequ = torch.cat((token_sequ, token_next), dim=1)
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens
#------

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # increase the efficiency
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in  range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    # x, y = get_batch("train")

    model = LanguageModel()
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

    #opt
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters):
        print(i)
        if i % eval_interval == 0 or i == max_iters-1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()         #get grad
        optimizer.step()        #minus grad

    print("train over")
    start_idx = random.randint(0, len(val_data)-block_size)

    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    context[0, :] = val_data[start_idx: start_idx+block_size]
    context_str = decode(context[0].tolist())


    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
    real_next_tokens[0, :] = val_data[start_idx+block_size: start_idx+block_size+max_new_tokens]
    real_next_tokens_str = decode(real_next_tokens[0].tolist())

    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())


    print("context:\n",context_str)
    print("real_next:\n",real_next_tokens_str)
    print("generated:\n",generated_str)

main()
