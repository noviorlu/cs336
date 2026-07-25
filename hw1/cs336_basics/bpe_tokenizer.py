import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import time
import heapq
from tqdm import tqdm

from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    
    local_vocab: dict[str, int] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # An empty pattern would make re.split cut between every character
        parts = re.split(macro_special_pattern, chunk) if macro_special_pattern else [chunk]

        for part in parts:
            for m in re.finditer(PAT, part):
                local_vocab[m.group()] += 1

    return local_vocab

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    macro_special_pattern = "|".join(re.escape(t) for t in special_tokens)

    result_bpe_vocab : dict[int, bytes] = {}
    for i in range(256):
        result_bpe_vocab[i] = bytes([i])
    result_bpe_merges : list[tuple[bytes, bytes]] = []

# region 1. chunk boundaries
    global_vocab: dict[str, int] = Counter()
    t0 = time.perf_counter()

    num_processes = os.cpu_count() or 1
    if special_tokens:
        # Split on a real special token so no chunk boundary can cut a pre-token in half
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
    else:
        boundaries = [0, os.path.getsize(input_path)]

    t1 = time.perf_counter()
    print(f"\nFinding chunk boundaries took {t1 - t0:.5f} seconds")
# endregion

# region 2. pre-tokenization
    t0 = time.perf_counter()
    with Pool(num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            results.append(
                pool.apply_async(
                    pretokenize_chunk, 
                    (input_path, start, end, macro_special_pattern)
                )
            )

        for result in results:
            local_vocab = result.get()
            global_vocab.update(local_vocab)

    t1 = time.perf_counter()
    print(f"Pre-tokenization took {t1 - t0:.5f} seconds")
# endregion

# region 3. BPE training
    t0 = time.perf_counter()

    global_vocab_ids :list[list[int]] = []                                      # 每个词的 id 序列（原地改）—— 注意去掉 count 单独存
    global_vocab_counts : list[int] = []                                        # 每个词的出现次数（和 word_freq 下标对齐）
    global_pair_counts: dict[tuple[int,int], int] = Counter()                   # 每个对的全局加权计数
    global_pair_to_words: dict[tuple[int,int], set[int]] = {}                   # 每个对 → 含它的词下标集合
    for idx, (token, count) in enumerate(global_vocab.items()):
        global_vocab_ids.append(list(token.encode("utf-8")))
        global_vocab_counts.append(count)

        ids_list = global_vocab_ids[-1]
        ids_length1 = len(ids_list) - 1
        for i in range(ids_length1):
            pair = (ids_list[i], ids_list[i + 1])
            global_pair_counts[pair] += count
            global_pair_to_words.setdefault(pair, set()).add(idx)

    pair_heap = PairHeap(result_bpe_vocab)
    for pair, count in global_pair_counts.items():
        pair_heap.push(pair, count)


    while len(result_bpe_vocab) < vocab_size - len(special_tokens):
        best_pair = pair_heap.pop_best(global_pair_counts)
        if best_pair is None: break
        id_a, id_b = best_pair
        new_id = len(result_bpe_vocab)
        result_bpe_vocab[new_id] = result_bpe_vocab[id_a] + result_bpe_vocab[id_b]
        result_bpe_merges.append((result_bpe_vocab[id_a], result_bpe_vocab[id_b]))


        affected = list(global_pair_to_words.get(best_pair, ()))
        for word_idx in affected:
            seq = global_vocab_ids[word_idx]
            seq_count = global_vocab_counts[word_idx]

            # Update the word's sequence by merging the best pair
            new_seq = []
            i = 0
            n = len(seq)
            while i < n:
                if i < n - 1 and seq[i] == id_a and seq[i+1] == id_b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            global_vocab_ids[word_idx] = new_seq

            # Only touch the pairs whose multiplicity in this word actually changed;
            # pairs left untouched keep their count, so their heap entry stays valid.
            old_pairs = Counter(zip(seq, seq[1:]))
            new_pairs = Counter(zip(new_seq, new_seq[1:]))
            for pair in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs[pair] - old_pairs[pair]
                if delta:
                    count = global_pair_counts[pair] + delta * seq_count
                    if count <= 0:
                        global_pair_counts.pop(pair, None)
                    else:
                        global_pair_counts[pair] = count
                        pair_heap.push(pair, count)

                if new_pairs[pair]:
                    global_pair_to_words.setdefault(pair, set()).add(word_idx)
                else:
                    words = global_pair_to_words.get(pair)
                    if words is not None:
                        words.discard(word_idx)
                        if not words:
                            del global_pair_to_words[pair]

        global_pair_counts.pop(best_pair, None)
        global_pair_to_words.pop(best_pair, None)
        pair_heap.maybe_rebuild(global_pair_counts)

    t1 = time.perf_counter()
    print(f"BPE Merge tooks {t1 - t0:.5f} seconds")
# endregion



    # Add special tokens to the vocabulary
    for token in special_tokens:
        result_bpe_vocab[len(result_bpe_vocab)] = token.encode("utf-8")

    return result_bpe_vocab, result_bpe_merges
