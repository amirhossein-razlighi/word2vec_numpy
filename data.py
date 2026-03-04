"""
Pipeline
--------
1.  Download (or load) a raw text corpus.
2.  Tokenise -> list of word strings.
3.  Build a vocabulary: word -> (index, frequency).
4.  Discard rare words (min_count).
5.  Subsample frequent words.
6.  Build the negative-sampling distribution table.
7.  Yield (center_idx, context_idx, neg_indices) training triples.

Dataset
-------
We use the *text8* corpus (first 10^8 characters of a cleaned Wikipedia
dump, ~17 M tokens after tokenisation).  It is ~30 MB and downloads
automatically on first run.
"""

import os
import re
import urllib.request
import zipfile
from collections import Counter
from typing import Iterator, List, Tuple

import numpy as np

TEXT8_URL  = "http://mattmahoney.net/dc/text8.zip"
TEXT8_PATH = "data/text8.zip"
TEXT8_FILE = "data/text8"


def download_text8(dest_dir: str = "data") -> str:
    """
    Download and unzip text8 into *dest_dir* if not already present.
    Returns the path to the unzipped text8 file.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_path  = os.path.join(dest_dir, "text8.zip")
    text_path = os.path.join(dest_dir, "text8")

    if not os.path.exists(text_path):
        if not os.path.exists(zip_path):
            print(f"Downloading text8 corpus (~30 MB) from {TEXT8_URL} …")
            urllib.request.urlretrieve(TEXT8_URL, zip_path)
            print("Download complete.")
        print("Unzipping …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"text8 saved to {text_path}")

    return text_path


def load_text8(path: str, max_tokens: int | None = None) -> List[str]:
    """
    Read the text8 file and return a flat list of lowercase word tokens.
    text8 is already a single long line of space-separated lowercase words.

    Parameters
    ----------
    max_tokens : int | None
        If set, truncate to the first *max_tokens* tokens (useful for quick
        experiments without using all 17 M tokens).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    tokens = raw.split()
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


class Vocabulary:
    """
    Maps words <-> integer indices and stores per-word frequencies.

    After construction:
      self.word2idx : dict[str, int]
      self.idx2word : list[str]
      self.freqs    : np.ndarray  (V,)  raw counts (aligned with indices)
    """

    def __init__(self, tokens: List[str], min_count: int = 5) -> None:
        counts = Counter(tokens)

        # Keep only words that appear at least min_count times.
        # Sort by descending frequency so the most common words get small indices.
        vocab_items = sorted(
            [(w, c) for w, c in counts.items() if c >= min_count],
            key=lambda x: -x[1],
        )

        self.word2idx: dict[str, int] = {w: i for i, (w, _) in enumerate(vocab_items)}
        self.idx2word: list[str]      = [w for w, _ in vocab_items]
        self.freqs:    np.ndarray     = np.array([c for _, c in vocab_items],
                                                 dtype=np.float64)

        # Normalised frequencies (probabilities) — used for subsampling.
        self.probs = self.freqs / self.freqs.sum()

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to indices; unknown words are dropped."""
        return [self.word2idx[w] for w in tokens if w in self.word2idx]


def subsample_mask(
    token_ids: List[int],
    vocab:     Vocabulary,
    t:         float = 1e-5,
    rng:       np.random.Generator | None = None,
) -> List[int]:
    """
    Stochastic down-sampling of frequent words.

    A word w is *discarded* with probability:

        P_discard(w) = 1 - sqrt( t / f(w) )

    where f(w) is the word's relative frequency and t is a threshold
    (typically 1e-4 ... 1e-5).  High-frequency words like "the", "of", "a"
    have P_discard close to 1 — they are seen far more often than needed.

    Equivalently, we *keep* the word with probability:

        P_keep = sqrt( t / f(w) ).

    Returns the filtered list of token ids.
    """
    if rng is None:
        rng = np.random.default_rng()

    # P_keep for every word index
    p_keep = np.sqrt(t / (vocab.probs + 1e-12))   # shape (V,)
    p_keep = np.minimum(p_keep, 1.0)

    # For each token draw a uniform random number; keep if u < p_keep[token]
    u = rng.random(len(token_ids))
    return [
        tok
        for tok, ui in zip(token_ids, u)
        if ui < p_keep[tok]
    ]


def build_negative_table(
    vocab:      Vocabulary,
    table_size: int = 10_000_000,
) -> np.ndarray:
    """
    Build an integer lookup table for fast negative sampling.

    Each word index w fills the table proportionally to

        freq(w)^{3/4}

    The 3/4 (= 0.75) exponent was chosen empirically by Mikolov et al.;
    it "smooths" the distribution — rare words are sampled more often
    than under the raw unigram and common words less often.

    Sampling a negative: pick a random integer in [0, table_size) and
    look up the word index stored there.  This is O(1) per sample.

    Parameters
    ----------
    table_size : int
        Larger -> more accurate approximation of the target distribution.
        10 M is the value used in the original word2vec C code.
    """
    # Smoothed unnormalised counts
    powered = vocab.freqs ** 0.75                # shape (V,)
    normed  = powered / powered.sum()            # normalise to a distribution

    # Number of table slots allocated to each word
    counts = (normed * table_size).astype(np.int64)

    # Build the table by repeating each index according to its slot count.
    table = np.repeat(np.arange(len(vocab), dtype=np.int32), counts)

    # The table may be slightly shorter/longer than table_size due to
    # integer rounding — trim or pad to exact size.
    if len(table) < table_size:
        table = np.concatenate([table,
                                np.zeros(table_size - len(table), dtype=np.int32)])
    else:
        table = table[:table_size]

    return table


def generate_pairs(
    token_ids:  List[int],
    window:     int,
    neg_table:  np.ndarray,
    n_negatives: int,
    rng:        np.random.Generator,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Yield (center_idx, context_idx, neg_indices) training triples.

    Skip-gram: for each token at position i (the "center"), sample a
    context window of size 1 ... window uniformly (dynamic window)
    and pair the center with each context token.

    Dynamic window: the actual window size is sampled uniformly from
    [1, window] for each center word.  This gives closer words a higher
    effective probability of being chosen as context.

    Parameters
    ----------
    token_ids   : encoded & subsampled corpus
    window      : maximum context window radius
    neg_table   : pre-built negative-sampling lookup table
    n_negatives : K — number of negative samples per positive pair
    rng         : numpy Generator for reproducibility
    """
    n = len(token_ids)
    table_size = len(neg_table)

    for i, center in enumerate(token_ids):
        # Dynamic window: sample actual radius r \in {1, ..., window}
        r = rng.integers(1, window + 1)

        lo = max(0, i - r)
        hi = min(n - 1, i + r)

        for j in range(lo, hi + 1):
            if j == i:
                continue                  # skip the center word itself

            context = token_ids[j]

            # Sample K negative indices from the noise distribution.
            neg_pos = rng.integers(0, table_size, size=n_negatives)
            neg_indices = neg_table[neg_pos]   # (K,)

            yield center, context, neg_indices


def prepare_data(
    corpus_path: str | None = None,
    max_tokens:  int  | None = 2_000_000,
    min_count:   int  = 5,
    subsample_t: float = 1e-5,
    neg_table_size: int = 10_000_000,
    seed:        int  = 42,
) -> Tuple[List[int], Vocabulary, np.ndarray]:
    """
    Full data-preparation pipeline.

    1. Download text8 if no corpus_path is given.
    2. Tokenise.
    3. Build vocabulary (min_count filtering).
    4. Encode corpus to integer ids.
    5. Subsample frequent words.
    6. Build negative-sampling table.

    Returns
    -------
    token_ids : List[int]       encoded, subsampled corpus
    vocab     : Vocabulary      word ↔ index mapping + frequencies
    neg_table : np.ndarray (T,) negative-sampling lookup table
    """
    rng = np.random.default_rng(seed)

    # --- Corpus loading ---------------------------------------------------
    if corpus_path is None:
        corpus_path = download_text8()

    print(f"Loading corpus from {corpus_path}...")
    tokens = load_text8(corpus_path, max_tokens=max_tokens)
    print(f"  Raw tokens : {len(tokens):,}")

    # --- Vocabulary -------------------------------------------------------
    vocab = Vocabulary(tokens, min_count=min_count)
    print(f"  Vocabulary : {len(vocab):,} words (min_count={min_count})")

    # --- Encode -----------------------------------------------------------
    token_ids = vocab.encode(tokens)
    print(f"  Encoded    : {len(token_ids):,} tokens after OOV removal")

    # --- Subsample --------------------------------------------------------
    token_ids = subsample_mask(token_ids, vocab, t=subsample_t, rng=rng)
    print(f"  After sub-sampling : {len(token_ids):,} tokens (t={subsample_t})")

    # --- Negative-sampling table ------------------------------------------
    print(f"  Building negative-sampling table (size={neg_table_size:,}) …")
    neg_table = build_negative_table(vocab, table_size=neg_table_size)

    return token_ids, vocab, neg_table
