import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


class BigramLM(nn.Module):

    def __init__(self, vocab_len):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits  = self.token_emb_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T) # (B*T := -1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, 1+1)
        return idx
    

if __name__ == "__main__":
    with open("input.txt", encoding="utf-8") as f:
        txt = f.read()
    
    chrs = sorted(list(set(txt)))
    vocab_len = len(chrs)

    stoi = { ch:i for i, ch in enumerate(chrs) }
    itos = { i:ch for i, ch in enumerate(chrs) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(txt), dtype=torch.long)

    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data   = data[n:]

    block_size = 8 # context_len
    train_data[:block_size+1]

    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"Input := {context}, Target := {target}")
    
    torch.manual_seed(1337)
    batch_size = 4

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    xb, yb = get_batch("train")

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"Input := {context}, Target := {target}")
    
    model = BigramLM(vocab_len)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3) # 3e-4 better for real network

    bs = 32
    losses = []
    for step in range(10000):
        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(loss.item())
    
    print("Final Bigram Loss:", loss.item())
    plt.plot(losses)

    idx = torch.zeros((1, 1), dtype=torch.long)
    pred = model.generate(idx = idx, max_new_tokens=300)[0].tolist()
    print(decode(pred))