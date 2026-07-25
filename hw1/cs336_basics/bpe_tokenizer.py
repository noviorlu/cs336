import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import time
from tqdm import tqdm

from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

def pretokenize(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_pattern: str,
) -> dict[str, int]:
    
    local_vocab: dict[str, int] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        parts = re.split(special_pattern, chunk)
        
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

    special_pattern = "|".join(re.escape(t) for t in special_tokens)

    bpe_vocab : dict[int, bytes] = {}
    for i in range(256):
        bpe_vocab[i] = bytes([i])
    bpe_merges : list[tuple[bytes, bytes]] = []

    global_vocab: dict[str, int] = Counter()
    
    t0 = time.perf_counter()

    boundaries = []
    num_processes = os.cpu_count() or 1
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    t1 = time.perf_counter()
    print(f"Finding chunk boundaries took {t1 - t0:.5f} seconds")

    with Pool(num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            results.append(
                pool.apply_async(
                    pretokenize, 
                    (input_path, start, end, special_pattern)
                )
            )

        for result in results:
            local_vocab = result.get()
            global_vocab.update(local_vocab)

    t2 = time.perf_counter()
    print(f"Pre-tokenization took {t2 - t1:.5f} seconds")

    word_freq : list[tuple[int, list[int]]] = []
    for token, count in global_vocab.items():
        word_freq.append((count, list(token.encode("utf-8"))))

    def rank(x):
        pair = x[0]
        count = x[1]
        id_a = pair[0]
        id_b = pair[1]
        bytes_a = bpe_vocab[id_a]
        bytes_b = bpe_vocab[id_b]
        return (count, (bytes_a, bytes_b))


    while len(bpe_vocab) < vocab_size - len(special_tokens):
        pairs: dict[tuple[int, int], int] = Counter()
        for count, token_id in word_freq:
            for i in range(len(token_id) - 1):
                pairs[(token_id[i], token_id[i + 1])] += count

        if not pairs:
            break  # No more pairs to merge

        # best_pair = Find pair in counts with MAX frequency (if tie, choose by lexicographically)
        best_pair, _ = max(pairs.items(), key=rank)
        id_a, id_b = best_pair
        new_id = len(bpe_vocab)
        bpe_vocab[new_id] = bpe_vocab[id_a] + bpe_vocab[id_b]
        bpe_merges.append((bpe_vocab[id_a], bpe_vocab[id_b]))

        # Update word_freq with new merged token
        for j, (count, token_id) in enumerate(word_freq):
            if id_a not in token_id:
                continue
            
            new_token_ids: list[int] = []
            i = 0
            token_length = len(token_id)
            token_length1 = len(token_id) - 1
            while i < token_length:
                if i < token_length1 and token_id[i] == id_a and token_id[i + 1] == id_b:
                    new_token_ids.append(new_id)  # New merged token ID
                    i += 2  # Skip the next byte since it's merged
                else:
                    new_token_ids.append(token_id[i])
                    i += 1
            
            word_freq[j] = (count, new_token_ids)

    # Add special tokens to the vocabulary
    for token in special_tokens:
        bpe_vocab[len(bpe_vocab)] = token.encode("utf-8")

    return bpe_vocab, bpe_merges
