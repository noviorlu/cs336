import time
from bpe_tokenizer import train_bpe
vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])