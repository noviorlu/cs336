"""Train and serialize the two tokenizers the assignment needs.

    python cs336_basics/main_bpe_train.py            # both
    python cs336_basics/main_bpe_train.py tinystories
    python cs336_basics/main_bpe_train.py owt

Outputs land in data/ as <name>_vocab.json + <name>_merges.json, readable back with
`load_tokenizer` / `Tokenizer.from_files`.

Also reports what §2.5 asks for: wall time, peak memory, and the longest token.
"""

import sys
import time
import resource
from pathlib import Path

try:
    from bpe_tokenizer import BPETrainer, store_tokenizer
except ImportError:  # also works when run from the repo root as a module
    from cs336_basics.bpe_tokenizer import BPETrainer, store_tokenizer

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

# name -> (corpus, vocab_size);  the sizes are fixed by the handout
CONFIGS = {
    "tinystories": ("TinyStoriesV2-GPT4-train.txt", 10_000),
    "owt": ("owt_train.txt", 32_000),
}
SPECIAL_TOKENS = ["<|endoftext|>"]


def peak_memory_gb() -> float:
    """Peak RSS of this process and its largest child, in GB.

    Pre-tokenization runs in workers, so the parent's own high-water mark misses them;
    ru_maxrss over RUSAGE_CHILDREN is the biggest single child, not their sum.
    """
    kb = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    return kb / 1024 / 1024


def run(name: str) -> None:
    corpus_name, vocab_size = CONFIGS[name]
    corpus = DATA / corpus_name
    if not corpus.exists():
        print(f"[{name}] skipped: {corpus} not found")
        return

    vocab_path = DATA / f"{name}_vocab.json"
    merges_path = DATA / f"{name}_merges.json"

    print(f"\n{'=' * 70}")
    print(f"[{name}] {corpus_name} ({corpus.stat().st_size / 1e9:.2f} GB), vocab_size={vocab_size}")
    print("=" * 70)

    # BPETrainer rather than train_bpe: we want the trainer's internals afterwards
    trainer = BPETrainer(vocab_size, SPECIAL_TOKENS)
    t0 = time.perf_counter()
    vocab, merges = trainer.train(str(corpus))
    elapsed = time.perf_counter() - t0

    store_tokenizer(vocab, merges, vocab_path, merges_path)

    # §2.5 deliverables
    longest = max(vocab.values(), key=len)
    print(f"\n[{name}] total {elapsed:.1f}s ({elapsed / 60:.1f} min), "
          f"peak memory {peak_memory_gb():.1f} GB")
    print(f"[{name}] vocab={len(vocab)}, merges={len(merges)}")
    # Free: the merge loop already left every pre-token in its encoded form (see
    # BPETrainer.training_compression_ratio). NOT the §2.7(a) number, which is measured
    # on 10 sampled documents and needs Tokenizer.encode.
    print(f"[{name}] compression ratio on the training corpus: "
          f"{trainer.training_compression_ratio():.3f} bytes/token")
    print(f"[{name}] longest token: {longest!r} ({len(longest)} bytes)")
    print(f"[{name}] wrote {vocab_path.name} ({vocab_path.stat().st_size / 1e6:.1f} MB)"
          f" + {merges_path.name} ({merges_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    # The guard matters: train_bpe spawns a multiprocessing Pool, and without it a
    # non-fork start method would re-import this file in every worker and recurse.
    targets = sys.argv[1:] or list(CONFIGS)
    unknown = [t for t in targets if t not in CONFIGS]
    if unknown:
        sys.exit(f"unknown target(s) {unknown}; choose from {list(CONFIGS)}")
    for target in targets:
        run(target)
