"""
train.py — Training loop for Skip-gram Word2Vec with Negative Sampling
========================================================================

Optimiser: Stochastic Gradient Descent (SGD) with a linearly decaying
learning rate.

Learning-rate schedule
----------------------
The original word2vec C code uses a simple linear decay:

    lr(t) = lr_0 x max( 1 - t / T,  lr_min_ratio )

where
  t   = number of training examples processed so far
  T   = total number of training examples across all epochs
  lr_min_ratio = 0.0001  (never drop below 0.01% of lr_0)

This is a "one-epoch linear warmdown" — it starts at lr_0, falls
linearly to near zero by the end of training.

Why SGD and not Adam?
---------------------
Word2Vec updates are very sparse (only the embeddings of the ~K+2
active words are touched per step).  Adam accumulates dense second-
moment estimates for *every* parameter which would require O(VxD) extra
memory and per-step work even for untouched embeddings.  SGD is both
memory-efficient and fast here.

Batch semantics
---------------
Each "batch" here is one (center, context, neg_indices) tuple — i.e.
batch_size = 1. For a pure-NumPy implementation with vectorised 
inner products the overhead is acceptable.
"""

import time
from typing import List, Tuple

import numpy as np

from model import Word2Vec
from data import Vocabulary, generate_pairs


def linear_decay_lr(
    lr_0:           float,
    step:           int,
    total_steps:    int,
    min_lr_ratio:   float = 1e-4,
) -> float:
    """
    Linearly decay lr from lr_0 → lr_0 x min_lr_ratio over total_steps.

    lr(t) = lr_0 x max(1 - t/T,  min_lr_ratio)
    """
    fraction = max(1.0 - step / total_steps, min_lr_ratio)
    return lr_0 * fraction


def train(
    model:       Word2Vec,
    token_ids:   List[int],
    vocab:       Vocabulary,
    neg_table:   np.ndarray,
    *,
    n_epochs:    int   = 5,
    window:      int   = 5,
    n_negatives: int   = 5,
    lr_0:        float = 0.025,
    log_every:   int   = 100_000,
    seed:        int   = 42,
) -> List[float]:
    """
    Train the Word2Vec model in-place and return per-log-step average losses.

    Parameters
    ----------
    model       : Word2Vec  (modified in-place)
    token_ids   : encoded + subsampled corpus
    vocab       : Vocabulary (for reporting only)
    neg_table   : negative-sampling lookup table
    n_epochs    : number of full passes over the corpus
    window      : maximum context-window radius
    n_negatives : K — number of negative samples per positive pair
    lr_0        : initial learning rate (0.025 is the C word2vec default)
    log_every   : print a progress line every this many training pairs
    seed        : RNG seed

    Returns
    -------
    loss_history : list of average losses recorded at each log checkpoint
    """
    rng = np.random.default_rng(seed)

    # Estimate total number of training pairs for the LR schedule.
    # Each token generates at most 2×window context pairs on average
    approx_pairs_per_epoch = len(token_ids) * 2 * window
    total_pairs = approx_pairs_per_epoch * n_epochs

    step         = 0       # global training-pair counter
    loss_history = []

    print(f"\n{'='*60}")
    print(f"  Training Word2Vec (Skip-gram + NEG)")
    print(f"  corpus tokens : {len(token_ids):,}")
    print(f"  vocab size    : {len(vocab):,}")
    print(f"  embed dim     : {model.embed_dim}")
    print(f"  epochs        : {n_epochs}")
    print(f"  window        : {window}")
    print(f"  negatives (K) : {n_negatives}")
    print(f"  lr_0          : {lr_0}")
    print(f"{'='*60}\n")

    for epoch in range(1, n_epochs + 1):
        epoch_loss   = 0.0
        epoch_pairs  = 0
        running_loss = 0.0
        t0 = time.time()

        pair_gen = generate_pairs(
            token_ids, window, neg_table, n_negatives, rng
        )

        for center, context, neg_indices in pair_gen:
            # ── learning-rate decay ───────────────────────────────────
            lr = linear_decay_lr(lr_0, step, total_pairs)

            # ── forward + backward + SGD update ──────────────────────
            loss = model.train_pair(center, context, neg_indices, lr)

            running_loss += loss
            epoch_loss   += loss
            epoch_pairs  += 1
            step         += 1

            # ── periodic progress report ──────────────────────────────
            if step % log_every == 0:
                avg = running_loss / log_every
                loss_history.append(avg)
                elapsed = time.time() - t0
                kpairs_per_sec = log_every / (elapsed + 1e-9) / 1000
                print(
                    f"  epoch {epoch}/{n_epochs}  "
                    f"step {step:>10,}  "
                    f"lr {lr:.6f}  "
                    f"avg_loss {avg:.4f}  "
                    f"{kpairs_per_sec:.1f} k pairs/s"
                )
                running_loss = 0.0
                t0 = time.time()

        epoch_avg = epoch_loss / max(epoch_pairs, 1)
        print(f"\n  ── Epoch {epoch} done | avg_loss {epoch_avg:.4f} "
              f"| pairs {epoch_pairs:,}\n")

    return loss_history


def save_vectors(path: str, model: Word2Vec, vocab: Vocabulary) -> None:
    """
    Save word vectors in word2vec text format:
        <vocab_size> <dim>
        <word> <v_0> <v_1> … <v_{D-1}>
    """
    V, D = model.vectors.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{V} {D}\n")
        for idx, word in enumerate(vocab.idx2word):
            vec_str = " ".join(f"{x:.6f}" for x in model.vectors[idx])
            f.write(f"{word} {vec_str}\n")
    print(f"Vectors saved to {path}  ({V} words x {D} dims)")


def load_vectors(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load vectors from the text format written by save_vectors().

    Returns
    -------
    vectors  : np.ndarray (V, D)
    idx2word : list[str]
    """
    vectors  = []
    idx2word = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        V, D = int(header[0]), int(header[1])
        for line in f:
            parts = line.rstrip().split(" ")
            idx2word.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    return np.array(vectors, dtype=np.float32), idx2word
