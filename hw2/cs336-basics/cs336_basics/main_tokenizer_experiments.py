"""handout §2.7 (a)(b)(c): compression ratio, cross-tokenizer, throughput.

    python cs336_basics/main_tokenizer_experiments.py
"""

import sys
import time
import random
from pathlib import Path

try:
    from bpe_tokenizer import Tokenizer, load_tokenizer
except ImportError:
    from cs336_basics.bpe_tokenizer import Tokenizer, load_tokenizer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
EOT = "<|endoftext|>"
N_DOCS = 10
SEED = 0

CORPORA = {
    "tinystories": "TinyStoriesV2-GPT4-valid.txt",
    "owt": "owt_valid.txt",
}


def sample_docs(path: Path, n: int, read_bytes: int = 64 << 20) -> list[str]:
    """Take the first `read_bytes` of the file, split on <|endoftext|>, sample n documents."""
    with open(path, "rb") as f:
        text = f.read(read_bytes).decode("utf-8", errors="ignore")
    docs = [d for d in text.split(EOT) if d.strip()]
    docs = docs[:-1]                       # last one is likely truncated by read_bytes
    return random.Random(SEED).sample(docs, min(n, len(docs)))


def compression_ratio(tok: Tokenizer, docs: list[str]) -> tuple[float, int, int]:
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_tokens = sum(len(tok.encode(d)) for d in docs)
    return n_bytes / n_tokens, n_bytes, n_tokens


def main() -> None:
    tokenizers = {
        name: Tokenizer.from_files(DATA / f"{name}_vocab.json", DATA / f"{name}_merges.json", [EOT])
        for name in CORPORA
    }
    samples = {name: sample_docs(DATA / fn, N_DOCS) for name, fn in CORPORA.items()}

    print("=" * 72)
    print("(a) 各用自己的 tokenizer 编码本语料的 10 篇文档")
    print("=" * 72)
    for name in CORPORA:
        ratio, nb, nt = compression_ratio(tokenizers[name], samples[name])
        print(f"  {name:12s} vocab={len(tokenizers[name].vocab):6d}  "
              f"{nb:8d} bytes / {nt:7d} tokens = {ratio:.3f} bytes/token")

    print()
    print("=" * 72)
    print("(b) 交叉：用 TinyStories 的 tokenizer 编码 OWT 样本")
    print("=" * 72)
    own, _, own_t = compression_ratio(tokenizers["owt"], samples["owt"])
    cross, _, cross_t = compression_ratio(tokenizers["tinystories"], samples["owt"])
    print(f"  OWT 样本 + OWT tokenizer(32k)          : {own:.3f} bytes/token  ({own_t} tokens)")
    print(f"  OWT 样本 + TinyStories tokenizer(10k)  : {cross:.3f} bytes/token  ({cross_t} tokens)")
    print(f"  -> 压缩率下降 {(1 - cross / own) * 100:.1f}%，token 数多 {cross_t / own_t - 1:+.1%}")

    print()
    print("=" * 72)
    print("(c) 吞吐（单进程，缓存冷启动）")
    print("=" * 72)
    for name, fn in CORPORA.items():
        with open(DATA / fn, "rb") as f:
            text = f.read(20 << 20).decode("utf-8", errors="ignore")
        tok = Tokenizer.from_files(                       # 新实例 = 空缓存
            DATA / f"{name}_vocab.json", DATA / f"{name}_merges.json", [EOT]
        )
        nb = len(text.encode("utf-8"))
        t0 = time.perf_counter()
        n_tokens = len(tok.encode(text))
        dt = time.perf_counter() - t0
        mbps = nb / 1e6 / dt
        pile_hours = 825e9 / (nb / dt) / 3600
        print(f"  {name:12s} {nb/1e6:6.1f} MB -> {n_tokens:9d} tokens in {dt:6.2f}s"
              f"  = {mbps:5.2f} MB/s")
        print(f"  {'':12s} Pile 825GB 单进程需 {pile_hours:8.1f} 小时 "
              f"({pile_hours/24:.1f} 天)，32 进程约 {pile_hours/32:.1f} 小时")


if __name__ == "__main__":
    main()
