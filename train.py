import torch

with open("./corpora/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# print(text[:1000])

chars = sorted(list(set(text)))
print("".join(chars))

# tokenizer: create a mapping between characters and integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s: str) -> list[int]: return [stoi[c] for c in s]
def decode(k: list[int]) -> str: return "".join([itos[i] for i in k])


print(encode("hi there"))
print(decode(encode("hi there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

