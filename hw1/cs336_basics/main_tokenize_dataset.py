"""handout §2.7(d): encode the datasets into uint16 token arrays on disk.

    python cs336_basics/main_tokenize_dataset.py                    # all four splits
    python cs336_basics/main_tokenize_dataset.py tinystories_valid
    python cs336_basics/main_tokenize_dataset.py owt_train owt_valid

Reads data/<name>_{vocab,merges}.json, writes data/<split>.npy (uint16).
That array is what §5 training reads back with np.memmap.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

try:
    from bpe_tokenizer import Tokenizer, find_chunk_boundaries
except ImportError:
    from cs336_basics.bpe_tokenizer import Tokenizer, find_chunk_boundaries

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
EOT = "<|endoftext|>"

# split -> (corpus file, which tokenizer)
SPLITS = {
    "tinystories_train": ("TinyStoriesV2-GPT4-train.txt", "tinystories"),
    "tinystories_valid": ("TinyStoriesV2-GPT4-valid.txt", "tinystories"),
    "owt_train": ("owt_train.txt", "owt"),
    "owt_valid": ("owt_valid.txt", "owt"),
}

# Set once per worker process. On Linux, Pool forks, so a tokenizer built here in the
# parent is inherited copy-on-write -- no need to ship vocab+merges with every task.
# The per-process pre-token cache also survives across that worker's tasks, which is
# why chunks are made larger than one-per-worker.
_TOKENIZER: Tokenizer | None = None
_CORPUS: Path | None = None


def _encode_chunk(start: int, end: int) -> np.ndarray:
    """Encode bytes [start, end) of the corpus. Runs in a worker process."""
    with open(_CORPUS, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    # uint16 rather than a list[int]: pickling the result back to the parent copies a
    # 2-bytes-per-token buffer instead of serialising tens of millions of Python ints.
    return np.fromiter(_TOKENIZER.encode(text), dtype=np.uint16)


def run(split: str, num_processes: int) -> None:
    global _TOKENIZER, _CORPUS
    corpus_name, tok_name = SPLITS[split]
    _CORPUS = DATA / corpus_name
    if not _CORPUS.exists():
        print(f"[{split}] skipped: {_CORPUS} not found")
        return

    _TOKENIZER = Tokenizer.from_files(
        DATA / f"{tok_name}_vocab.json", DATA / f"{tok_name}_merges.json", [EOT]
    )
    vocab_size = len(_TOKENIZER.vocab)
    # uint16 holds 0..65535. Every current token ID fits; a 100k+ vocab would not.
    assert vocab_size <= 65536, f"vocab_size={vocab_size} does not fit in uint16"

    out_path = DATA / f"{split}.npy"
    size = _CORPUS.stat().st_size
    print(f"\n{'=' * 70}")
    print(f"[{split}] {corpus_name} ({size / 1e9:.2f} GB), tokenizer={tok_name} (vocab={vocab_size})")
    print("=" * 70)

    t0 = time.perf_counter()
    # Boundaries land on <|endoftext|>, a hard boundary no pre-token or merge crosses,
    # so per-chunk encoding is identical to encoding the whole file.
    with open(_CORPUS, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, EOT.encode("utf-8"))
    spans = list(zip(boundaries[:-1], boundaries[1:]))

    with Pool(num_processes) as pool:
        results = [pool.apply_async(_encode_chunk, span) for span in spans]
        parts = [r.get() for r in results]        # 按块序收回，顺序即原文顺序
    ids = np.concatenate(parts)
    del parts
    encode_time = time.perf_counter() - t0

    np.save(out_path, ids)
    n = len(ids)
    print(f"[{split}] {len(spans)} chunks x {num_processes} procs, {encode_time:.1f}s "
          f"({size / 1e6 / encode_time:.1f} MB/s)")
    print(f"[{split}] {n:,} tokens, compression {size / n:.3f} bytes/token")
    print(f"[{split}] max id = {ids.max()} (< 65536 ✓), wrote {out_path.name} "
          f"({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(SPLITS)
    unknown = [t for t in targets if t not in SPLITS]
    if unknown:
        sys.exit(f"unknown split(s) {unknown}; choose from {list(SPLITS)}")
    nproc = os.cpu_count() or 1
    for target in targets:
        run(target, nproc)
