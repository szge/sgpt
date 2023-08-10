import torch
import torch.nn as nn
from torch.nn import functional as fn
import os
import tiktoken

# region hyperparameters
# full
# BATCH_SIZE = 16  # number of sequences per batch
# BLOCK_SIZE = 64  # length of sequence to process at a time
# MAX_ITERS = 5000
# EVAL_INTERVAL = 500
# LEARNING_RATE = 3e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EVAL_ITERS = 200
# N_EMBED = 60  # must be divisible by N_HEADS
# N_HEADS = 6
# N_LAYERS = 4
# DROPOUT = 0.2

# test
BATCH_SIZE = 32  # number of sequences per batch
BLOCK_SIZE = 8  # length of sequence to process at a time
MAX_ITERS = 3000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBED = 32  # must be divisible by N_HEADS
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.2
# endregion

torch.manual_seed(1337)

with open("./corpora/cntkillme.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokenizer: create a mapping between characters and integers
# chars = sorted(list(set(text)))
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}
# def encode(s: str) -> list[int]: return [stoi[c] for c in s]
# def decode(k: list[int]) -> str: return "".join([itos[i] for i in k])

# use tiktoken
enc = tiktoken.get_encoding("cl100k_base")
encode = enc.encode
decode = enc.decode
vocab_size = enc.max_token_value + 1

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        b, t, c = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * c ** (-0.5)  # normalize to produce variance 1
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = fn.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttn(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embed, n_head):
        """
        :param n_embed: embedding dimension
        :param n_head: number of heads we'd like
        """
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttn(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape

        # idx and targets are both (B, T) tensors of integers
        token_emb = self.token_embedding_table(idx)  # (B, T, C) batch, time, channel
        pos_emb = self.position_embedding_table(torch.arange(t, device=DEVICE))  # (T, C) time, channel
        x = token_emb + pos_emb  # (B, T, C) batch, time, channel
        x = self.blocks(x)  # (B, T, C) batch, time, channel
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V) batch, time, vocab_size

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = fn.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]  # crop idx to the last BLOCK_SIZE tokens
            logits, loss = self(idx_cond)  # get the prediction
            logits = logits[:, -1, :]  # focus only on the last step, becomes (B, C)
            probs = fn.softmax(logits, dim=-1)  # apply softmax to get probabilities, (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from the distribution, (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx


# check if model is saved
if os.path.exists("model.pt"):
    model = torch.load("model.pt")
    m = model.to(DEVICE)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
else:
    model = BigramLM()
    m = model.to(DEVICE)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for iteration in range(MAX_ITERS):
        # every once in a while evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    # save the model
    torch.save(model, "model.pt")
