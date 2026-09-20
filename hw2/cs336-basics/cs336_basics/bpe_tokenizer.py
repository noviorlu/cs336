"""BPE tokenizer: training (§2.4-2.5), serialization, and encode/decode (§2.6).

Layout
    1. Pre-tokenization helpers  -- module-level, because multiprocessing pickles by name
    2. PairHeap                  -- lazy-deletion max-heap over adjacent pairs
    3. BPETrainer / train_bpe    -- the training algorithm
    4. store/load_tokenizer      -- serialization, so a 5-minute owt run survives the process
    5. Tokenizer                 -- §2.6 encode/decode
"""

import os
import gc
import itertools
import json
import time
import heapq
import regex as re
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool
from collections import Counter

from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)


# ══════════════════════════════════════════════════════════════════════════════
# region 1. Pre-tokenization
# ══════════════════════════════════════════════════════════════════════════════

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    macro_special_pattern: str,
) -> dict[str, int]:
    """Count pre-tokens in [start, end) of the file. Runs in a worker process."""
    local_vocab: dict[str, int] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # An empty pattern would make re.split cut between every character
        parts = re.split(macro_special_pattern, chunk) if macro_special_pattern else [chunk]

        # findall + Counter.update counts in C; finditer + `+= 1` pays a Python
        # loop and a Match object per token. Documents are small, so the
        # intermediate list is cheap.
        for part in parts:
            local_vocab.update(COMPILED_PAT.findall(part))

    return local_vocab

# endregion 1. Pre-tokenization
# ══════════════════════════════════════════════════════════════════════════════
# region 2. PairHeap
# ══════════════════════════════════════════════════════════════════════════════

def neg_bytes(b: bytes) -> tuple:
    """Order-reversing key for bytes. The trailing sentinel (> any -x) reverses prefix order too."""
    return tuple(-x for x in b) + (1,)


class PairHeap:
    """Lazy-deletion max-heap over pairs, ordered by (count, (bytes_a, bytes_b)) descending.

    Stale entries are tolerated: every count change pushes a fresh entry, so each live pair
    always has an entry carrying its current count, and pop_best drops anything that
    disagrees with pair_counts.
    """

    def __init__(self, vocab):
        self.vocab = vocab
        self.heap = []
        self._neg_cache: dict[int, tuple] = {}

    def _neg(self, token_id: int) -> tuple:
        key = self._neg_cache.get(token_id)
        if key is None:
            key = self._neg_cache[token_id] = neg_bytes(self.vocab[token_id])
        return key

    def _key(self, pair, count):
        id_a, id_b = pair
        return (-count, self._neg(id_a), self._neg(id_b), pair)

    def push(self, pair, count):
        heapq.heappush(self.heap, self._key(pair, count))

    def pop_best(self, pair_counts):
        while self.heap:
            neg_count, _, _, pair = heapq.heappop(self.heap)
            count = -neg_count
            if count == pair_counts.get(pair, 0) and count > 0:
                return pair
        return None

    def maybe_rebuild(self, pair_counts):
        """Drop accumulated stale entries once they dominate the heap."""
        if len(self.heap) > 2 * len(pair_counts) + 1024:
            self.heap = [self._key(p, c) for p, c in pair_counts.items() if c > 0]
            heapq.heapify(self.heap)

# endregion 2. PairHeap
# ══════════════════════════════════════════════════════════════════════════════
# region 3. Training
# ══════════════════════════════════════════════════════════════════════════════

class BPETrainer:
    """Trains a byte-level BPE tokenizer.

    Split into phases so each can be timed/tested on its own:
        _chunk_boundaries -> _pretokenize -> _build_indexes -> _merge_loop

    The merge loop keeps every hot structure in a local variable rather than on
    `self`: an attribute lookup per inner-loop access is measurable at owt scale.
    """

    def __init__(self, vocab_size: int, special_tokens: list[str], verbose: bool = True):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.verbose = verbose
        self.macro_special_pattern = "|".join(re.escape(t) for t in special_tokens)

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []

        # filled in by _build_indexes
        self.word_ids: list[list[int]] = []                       # 每词的 id 序列（原地改）
        self.word_counts: list[int] = []                          # 每词出现次数（下标对齐）
        self.word_nbytes: list[int] = []                          # 每词的原始字节数（word_ids 会被改掉）
        self.word_pairs: list[dict[tuple[int, int], int]] = []    # 每词内部 pair→次数（常驻）
        self.pair_counts: dict[tuple[int, int], int] = Counter()  # pair→全局加权计数（真相源）
        self.pair_to_words: dict[tuple[int, int], set[int]] = {}  # pair→含它的词下标集合

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ─── phase 1 ──────────────────────────────────────────────────────────────
    def _chunk_boundaries(self, input_path, num_processes: int) -> list[int]:
        t0 = time.perf_counter()
        # More chunks than workers: documents are unevenly sized, so oversubscribing
        # lets fast workers pick up more tasks instead of idling on the slowest chunk
        desired_chunks = num_processes * 4
        if self.special_tokens:
            # Split on a real special token so no chunk boundary can cut a pre-token in half
            with open(input_path, "rb") as f:
                boundaries = find_chunk_boundaries(
                    f, desired_chunks, self.special_tokens[0].encode("utf-8")
                )
        else:
            boundaries = [0, os.path.getsize(input_path)]
        self._log(f"\nFinding chunk boundaries took {time.perf_counter() - t0:.5f} seconds")
        return boundaries

    # ─── phase 2 ──────────────────────────────────────────────────────────────
    def _pretokenize(self, input_path, boundaries, num_processes: int) -> Counter:
        t0 = time.perf_counter()
        word_freq: Counter = Counter()
        with Pool(num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
            results = [
                pool.apply_async(
                    pretokenize_chunk,
                    (input_path, start, end, self.macro_special_pattern),
                )
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            for result in results:
                word_freq.update(result.get())
        self._log(f"Pre-tokenization took {time.perf_counter() - t0:.5f} seconds")
        return word_freq

    # ─── phase 3 ──────────────────────────────────────────────────────────────
    def _build_indexes(self, word_freq: Counter):
        """One pass over the frequency table fills all four persistent structures."""
        pair_counts = self.pair_counts
        pair_to_words = self.pair_to_words
        for idx, (token, count) in enumerate(word_freq.items()):
            ids_list = list(token.encode("utf-8"))
            self.word_ids.append(ids_list)
            self.word_counts.append(count)
            self.word_nbytes.append(len(ids_list))   # 现在 1 id = 1 字节，之后就不是了

            wp: dict[tuple[int, int], int] = {}
            for i in range(len(ids_list) - 1):
                pair = (ids_list[i], ids_list[i + 1])
                pair_counts[pair] += count
                pair_to_words.setdefault(pair, set()).add(idx)
                wp[pair] = wp.get(pair, 0) + 1
            self.word_pairs.append(wp)

    # ─── phase 4 ──────────────────────────────────────────────────────────────
    def _merge_loop(self):
        t0 = time.perf_counter()

        # bind to locals: the inner loop runs ~10^8 times at owt scale
        vocab = self.vocab
        merges = self.merges
        word_ids = self.word_ids
        word_counts = self.word_counts
        all_word_pairs = self.word_pairs
        pair_counts = self.pair_counts
        pair_to_words = self.pair_to_words
        target = self.vocab_size - len(self.special_tokens)

        pair_heap = PairHeap(vocab)
        for pair, count in pair_counts.items():
            pair_heap.push(pair, count)

        delta: dict[tuple[int, int], int] = {}   # 复用同一个 dict，省掉每词一次分配
        while len(vocab) < target:
            best_pair = pair_heap.pop_best(pair_counts)
            if best_pair is None:
                break
            id_a, id_b = best_pair
            new_id = len(vocab)
            vocab[new_id] = vocab[id_a] + vocab[id_b]
            merges.append((vocab[id_a], vocab[id_b]))

            for word_idx in list(pair_to_words.get(best_pair, ())):
                seq = word_ids[word_idx]
                seq_count = word_counts[word_idx]

                # Single pass: rebuild the sequence AND record only the pairs that changed.
                # A pair straddling two untouched tokens is identical before and after, so it
                # never enters `delta` -- that is the whole point (the old code rebuilt two
                # full Counters per word just to discover most pairs were unchanged).
                new_seq = []
                i = 0
                n = len(seq)
                prev_new = -1          # last token appended to new_seq
                prev_old_end = -1      # old token that this element ends on
                prev_merged = False
                while i < n:
                    if i < n - 1 and seq[i] == id_a and seq[i + 1] == id_b:
                        cur_new = new_id
                        cur_old_start = id_a
                        cur_old_end = id_b
                        cur_merged = True
                        delta[best_pair] = delta.get(best_pair, 0) - 1
                        i += 2
                    else:
                        cur_new = cur_old_start = cur_old_end = seq[i]
                        cur_merged = False
                        i += 1

                    # A boundary pair changes iff one of the two sides was merged. Testing
                    # flags instead of comparing tuples keeps untouched positions allocation-free.
                    if prev_new >= 0 and (prev_merged or cur_merged):
                        old_pair = (prev_old_end, cur_old_start)
                        new_pair = (prev_new, cur_new)
                        delta[old_pair] = delta.get(old_pair, 0) - 1
                        delta[new_pair] = delta.get(new_pair, 0) + 1

                    new_seq.append(cur_new)
                    prev_new = cur_new
                    prev_old_end = cur_old_end
                    prev_merged = cur_merged

                word_ids[word_idx] = new_seq

                # The resident per-word pair counts turn a delta into an absolute membership
                # answer, which the neighbourhood scan alone cannot give.
                word_pairs = all_word_pairs[word_idx]
                for pair, d in delta.items():
                    if not d:
                        continue
                    new_in_word = word_pairs.get(pair, 0) + d
                    if new_in_word:
                        word_pairs[pair] = new_in_word
                    else:
                        del word_pairs[pair]

                    count = pair_counts[pair] + d * seq_count
                    if count <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = count
                        pair_heap.push(pair, count)

                    if new_in_word:
                        if new_in_word == d:      # 0 -> positive: newly introduced
                            pair_to_words.setdefault(pair, set()).add(word_idx)
                    else:
                        words = pair_to_words.get(pair)
                        if words is not None:
                            words.discard(word_idx)
                            if not words:
                                del pair_to_words[pair]
                delta.clear()

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            pair_heap.maybe_rebuild(pair_counts)

        self._log(f"BPE Merge tooks {time.perf_counter() - t0:.5f} seconds")

    def training_compression_ratio(self) -> float:
        """bytes / token over the *training* corpus. Call after `train`.

        No encoder needed: when the merge loop ends, `word_ids[i]` is the result of
        applying merges 1..N in creation order to pre-token i -- which is exactly what
        `Tokenizer.encode` does. The trainer has already encoded the whole corpus.

        Excludes the special-token delimiters, which pre-tokenization split away.
        """
        total_bytes = sum(c * n for c, n in zip(self.word_counts, self.word_nbytes))
        total_tokens = sum(c * len(ids) for c, ids in zip(self.word_counts, self.word_ids))
        return total_bytes / total_tokens

    # ─── driver ───────────────────────────────────────────────────────────────
    def train(self, input_path: str | os.PathLike) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        num_processes = os.cpu_count() or 1
        boundaries = self._chunk_boundaries(input_path, num_processes)
        word_freq = self._pretokenize(input_path, boundaries, num_processes)

        # The merge loop allocates millions of small containers (sequences, per-word pair
        # dicts, heap keys). None of them form reference cycles, so refcounting alone frees
        # them -- but the generational GC still rescans the whole live set on every gen-2
        # pass, which costs more than the loop itself. Measured on owt_valid: 42.8s -> 28.2s
        # before the neighbourhood-delta rewrite, 54.2s -> 19.2s after.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            self._build_indexes(word_freq)
            del word_freq
            self._merge_loop()
        finally:
            if gc_was_enabled:
                gc.enable()

        # Special tokens go last: they never participate in merges, they just occupy IDs
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        return self.vocab, self.merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Functional entry point kept for `adapters.run_train_bpe` and the tests."""
    return BPETrainer(vocab_size, special_tokens, verbose=kwargs.get("verbose", True)).train(input_path)


# endregion 3. Training
# ══════════════════════════════════════════════════════════════════════════════
# region 4. Serialization
# ══════════════════════════════════════════════════════════════════════════════
#
# Why latin-1 and not utf-8: a merged token is an arbitrary byte string and is NOT
# guaranteed to be valid UTF-8 (a merge can land in the middle of a multi-byte
# character). latin-1 maps bytes 0..255 one-to-one onto codepoints U+0000..U+00FF,
# so `b.decode("latin-1").encode("latin-1") == b` for *every* byte string.

def store_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Write vocab and merges to disk in the format `Tokenizer.from_files` reads back."""
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(i): b.decode("latin-1") for i, b in vocab.items()}, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(
            [[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges], f, ensure_ascii=False
        )


def load_tokenizer(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Inverse of `store_tokenizer`."""
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {int(i): s.encode("latin-1") for i, s in json.load(f).items()}
    with open(merges_path, encoding="utf-8") as f:
        merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in json.load(f)]
    return vocab, merges

def lookup_token_id(
    vocab_path: str | os.PathLike,
    token: str = "<|endoftext|>",
) -> int:
    """从 vocab.json 里查一个 token 的 id，查不到就报错。

    用途是让训练脚本不必手填文档分隔符的 id——它随词表变（TinyStories 10k 是
    9999，OWT 32k 是 31999），手填错了不会崩，只会静默生成一份错误的 document
    mask（那个 id 在另一套词表里往往是个普通 token，于是注意力被切在随机位置）。
    只读 vocab，不加载 merges。
    """
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    target = token.encode("utf-8").decode("latin-1")
    for i, s in vocab.items():
        if s == target:
            return int(i)
    raise ValueError(f"{vocab_path} 里没有 token {token!r}（词表大小 {len(vocab)}）")


# endregion 4. Serialization
# ══════════════════════════════════════════════════════════════════════════════
# region5. Tokenizer (§2.6)
# ══════════════════════════════════════════════════════════════════════════════

class Tokenizer:
    """Encode text to token IDs and decode them back, using a trained vocab + merges."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a vocabulary, a list of merges, and optional special tokens.

        Args:
            vocab: token ID -> token bytes.
            merges: BPE merges in creation order; index = priority (lower merges first).
            special_tokens: strings that must never be split. Append any that are not
                already in `vocab`.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.inverse_vocab: dict[bytes, int] = {b: i for i, b in vocab.items()}
        for token in self.special_tokens:
            b = token.encode("utf-8")
            if b not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = b
                self.inverse_vocab[b] = new_id

        self.merge_rank: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.cached_word_ids: dict[str, list[int]] = {}

        # Special tokens are split out BEFORE PAT runs: they are a hard boundary that the
        # regex must not look across (a lookahead peeking past one changes how the
        # whitespace around it is tokenised). Longest first -- alternation is
        # leftmost-first, not longest-match, so "<|eot|>" ahead of "<|eot|><|eot|>"
        # would cut the double one in half. The capture group keeps the delimiters.
        self.special_set: set[str] = set(self.special_tokens)
        self.special_split_pat = (
            re.compile(
                "(" + "|".join(
                    re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
                ) + ")"
            )
            if self.special_tokens
            else None
        )

    @classmethod
    def from_files(
        cls: type["Tokenizer"],
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from files produced by `store_tokenizer`."""
        vocab, merges = load_tokenizer(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        parts = self.special_split_pat.split(text) if self.special_split_pat else [text]
        for part in parts:
            if not part:                       # re.split emits empty strings at the edges
                continue
            if part in self.special_set:       # one ID straight from the table: no PAT, no merges
                ids.append(self.inverse_vocab[part.encode("utf-8")])
                continue
            for pre_token in COMPILED_PAT.findall(part):
                # Zipf: the same pre-token recurs constantly, so merge it once and reuse.
                # Keyed on str so a cache hit costs no re-encode.
                merged = self.cached_word_ids.get(pre_token)
                if merged is None:
                    merged = self.cached_word_ids[pre_token] = self._merge_pretoken(pre_token)
                ids.extend(merged)
        return ids

    def _merge_pretoken(self, pre_token: str) -> list[int]:
        seq = [bytes([b]) for b in pre_token.encode("utf-8")]
        rank = self.merge_rank

        while len(seq) > 1:
            # Only pairs that actually have a merge are candidates.
            candidates = [pair for pair in zip(seq, seq[1:]) if pair in rank]
            if not candidates:
                break
            a, b = min(candidates, key=rank.__getitem__)

            # Merge every occurrence, scanning left to right. Overlaps resolve greedily:
            # [a, a, a] with (a, a) becomes [aa, a] -- the same way training merged them,
            # which is what keeps encode consistent with the merge list it replays.
            merged_seq = []
            i = 0
            n = len(seq)
            while i < n:
                if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                    merged_seq.append(a + b)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            seq = merged_seq

        # Every byte string here is either one of the 256 base bytes or the product of a
        # merge, so it is guaranteed to be in the vocabulary.
        return [self.inverse_vocab[token] for token in seq]


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        S = max((len(s) for s in self.special_tokens), default=0)
        buf = ""
        pending = None
        for chunk in itertools.chain(iterable, [None]):
            if pending is None and chunk is not None:
                pending = chunk
                continue
            buf += pending or ""
            pending = chunk
            eof = chunk is None

            if eof:
                limit = len(buf)
            else:
                limit = len(buf) - (S - 1) if S > 1 else len(buf)
                if limit <= 0:
                    continue
                if self.special_split_pat:
                    for m in self.special_split_pat.finditer(buf):
                        if m.start() < limit < m.end():
                            limit = m.start()
                if limit <= 0:
                    continue

            # 在 [0, limit) 上按 special 边界切段：special 是硬边界，PAT 不能跨过它
            region = buf[:limit]
            pieces, cursor = [], 0
            if self.special_split_pat:
                for m in self.special_split_pat.finditer(region):
                    pieces.append((False, region[cursor:m.start()]))
                    pieces.append((True, m.group()))
                    cursor = m.end()
            pieces.append((False, region[cursor:]))

            held_len = 0
            for i, (is_special, piece) in enumerate(pieces):
                if not piece:
                    continue
                if is_special:
                    yield self.inverse_vocab[piece.encode("utf-8")]
                    continue
                pre_tokens = COMPILED_PAT.findall(piece)
                if i == len(pieces) - 1 and not eof and pre_tokens:
                    held_len = len(pre_tokens.pop())      # hazard 2: 扣住最后一个
                for pre_token in pre_tokens:
                    merged = self.cached_word_ids.get(pre_token)
                    if merged is None:
                        merged = self.cached_word_ids[pre_token] = self._merge_pretoken(pre_token)
                    yield from merged
            buf = buf[limit - held_len:]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text.

        IDs come from the user and are not guaranteed to form valid UTF-8; malformed
        bytes must become U+FFFD.
        """
        # Join first, decode once. Decoding token by token would be wrong: a merge can
        # land in the middle of a multi-byte character, so one token may hold only the
        # lead byte and the next token the continuation bytes.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
