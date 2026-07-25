import time
from bpe_tokenizer import train_bpe

start = time.perf_counter()
vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
elapsed = time.perf_counter() - start
print(f"训练耗时: {elapsed:.1f} 秒")
print(f"vocab 大小: {len(vocab)}")