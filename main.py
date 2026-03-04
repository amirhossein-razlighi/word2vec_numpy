"""
main.py — Entry point: configure, train, evaluate, and save
============================================================

Usage
-----
    # Quick smoke-test (2 M tokens, 3 epochs)
    python main.py

    # Full training on all 17 M tokens
    python main.py --max_tokens 0 --epochs 5

    # Custom corpus
    python main.py --corpus path/to/mytext.txt
"""

import argparse
import os

import numpy as np

from data    import prepare_data
from model   import Word2Vec
from train   import train, save_vectors, plot_training
from evaluate import EmbeddingSpace, print_nearest, print_analogies


DEFAULTS = dict(
    corpus      = None,           # auto-download text8
    max_tokens  = 2_000_000,      # set to 0 for the full corpus
    min_count   = 5,              # discard rare words
    subsample_t = 1e-5,           # aggressive subsampling threshold
    embed_dim   = 100,            # D — embedding dimensionality
    window      = 5,              # maximum context-window radius
    n_negatives = 5,              # K — negative samples per positive pair
    n_epochs    = 3,              # training epochs
    lr_0        = 0.025,          # initial SGD learning rate
    seed        = 42,             # global random seed
    vectors_out = "vectors.txt",  # path for saving trained vectors
    log_every   = 100_000,        # report every N training pairs
    plots_dir   = ".",            # directory for saving training-curve plots
)


def main(cfg: argparse.Namespace) -> None:
    rng = np.random.default_rng(cfg.seed)
    max_tok = cfg.max_tokens if cfg.max_tokens > 0 else None

    # ── 1. Data preparation ───────────────────────────────────────────────
    token_ids, vocab, neg_table = prepare_data(
        corpus_path    = cfg.corpus,
        max_tokens     = max_tok,
        min_count      = cfg.min_count,
        subsample_t    = cfg.subsample_t,
        neg_table_size = 10_000_000,
        seed           = cfg.seed,
    )

    # ── 2. Model initialisation ───────────────────────────────────────────
    model = Word2Vec(
        vocab_size = len(vocab),
        embed_dim  = cfg.embed_dim,
        seed       = cfg.seed,
    )

    # ── 3. Training ───────────────────────────────────────────────────────
    loss_history, lr_history = train(
        model,
        token_ids,
        vocab,
        neg_table,
        n_epochs    = cfg.n_epochs,
        window      = cfg.window,
        n_negatives = cfg.n_negatives,
        lr_0        = cfg.lr_0,
        log_every   = cfg.log_every,
        seed        = cfg.seed,
    )

    # ── 4. Save embeddings ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg.vectors_out) or ".", exist_ok=True)
    save_vectors(cfg.vectors_out, model, vocab)

    # ── 5. Plot training curves (requires matplotlib — skips if absent) ───
    out_png = os.path.join(cfg.plots_dir, "training_curves.png")
    plot_training(
        loss_history,
        lr_history,
        log_every = cfg.log_every,
        out_path  = out_png,
    )

    # ── 5. Intrinsic evaluation ───────────────────────────────────────────
    space = EmbeddingSpace(model.vectors, vocab.idx2word)

    # ---- Nearest neighbours ----
    probe_words = ["king", "paris", "computer", "music", "dog", "war"]
    print_nearest(space, probe_words, topn=8)

    # ---- Analogies (a : a* :: b : ?) ----
    analogies = [
        # Capital cities:
        ("france", "paris",   "germany"),    # -> berlin
        ("france", "paris",   "japan"),      # -> tokyo
        # Gender:
        ("king",   "man",     "queen"),      # -> woman
        ("actor",  "man",     "actress"),    # -> woman
        # Verb tense:
        ("walk",   "walking", "run"),        # -> running
        # Comparative:
        ("good",   "better",  "bad"),        # -> worse
    ]
    print_analogies(space, analogies)

    # ---- Final loss summary ----
    if loss_history:
        print(f"Loss trajectory (first -> last checkpoint): "
              f"{loss_history[0]:.4f} -> {loss_history[-1]:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train Skip-gram Word2Vec with Negative Sampling (pure NumPy)"
    )
    p.add_argument("--corpus",      type=str,   default=DEFAULTS["corpus"],
                   help="Path to a UTF-8 text file. Omit to auto-download text8.")
    p.add_argument("--max_tokens",  type=int,   default=DEFAULTS["max_tokens"],
                   help="Truncate corpus to this many tokens (0 = use all).")
    p.add_argument("--min_count",   type=int,   default=DEFAULTS["min_count"])
    p.add_argument("--subsample_t", type=float, default=DEFAULTS["subsample_t"])
    p.add_argument("--embed_dim",   type=int,   default=DEFAULTS["embed_dim"])
    p.add_argument("--window",      type=int,   default=DEFAULTS["window"])
    p.add_argument("--n_negatives", type=int,   default=DEFAULTS["n_negatives"])
    p.add_argument("--n_epochs",    type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--lr_0",        type=float, default=DEFAULTS["lr_0"])
    p.add_argument("--seed",        type=int,   default=DEFAULTS["seed"])
    p.add_argument("--vectors_out", type=str,   default=DEFAULTS["vectors_out"])
    p.add_argument("--log_every",   type=int,   default=DEFAULTS["log_every"])
    p.add_argument("--plots_dir",   type=str,   default=DEFAULTS["plots_dir"],
                   help="Directory to save training-curve plots (requires matplotlib).")

    args = p.parse_args()
    main(args)
