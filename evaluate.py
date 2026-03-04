"""
evaluate.py — Intrinsic evaluation for Word2Vec embeddings
===========================================================

Two standard intrinsic evaluations:

1. **Nearest neighbours** — sanity check that semantically related words
   cluster together in the embedding space.

2. **Word analogy** (3CosAdd) — the famous "king - man + woman ≈ queen"
    test of linear relationships in the embedding space.
    
Cosine similarity
-----------------
All comparisons use cosine similarity:

    cos(u, v) = (u · v) / (‖u‖ · ‖v‖)

Range: [-1, 1].  For normalised vectors it reduces to a dot product.

We pre-normalise the entire matrix once so that batch queries are a
single matrix-vector multiplication:

    scores = E_norm @ query_norm      (V,)

This is O(V x D) per query.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np


class EmbeddingSpace:
    """
    Wraps word vectors with fast cosine-similarity lookups.

    Parameters
    ----------
    vectors  : np.ndarray (V, D)   — raw (unnormalised) word vectors
    idx2word : list[str]           — index-to-word mapping
    """

    def __init__(self, vectors: np.ndarray, idx2word: List[str]) -> None:
        self.vectors  = vectors.astype(np.float32)
        self.idx2word = idx2word
        self.word2idx = {w: i for i, w in enumerate(idx2word)}

        # Pre-normalise rows for fast cosine via dot product.
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)          # avoid division by zero
        self.normed = self.vectors / norms         # (V, D)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __len__(self) -> int:
        return len(self.idx2word)

    def vec(self, word: str) -> np.ndarray:
        """Return the (unnormalised) vector for *word*."""
        return self.vectors[self.word2idx[word]]

    def vec_norm(self, word: str) -> np.ndarray:
        """Return the L2-normalised vector for *word*."""
        return self.normed[self.word2idx[word]]

    def most_similar(
        self,
        word:    str,
        top_k:    int = 10,
        exclude: List[str] | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Return the *top_k* words most cosine-similar to *word*.

        Strategy
        --------
        1. Compute cosine scores for all vocabulary words in one
           matrix-vector product: scores = E_norm @ q_norm   (V,)
        2. Use np.argpartition to get the top-(top_k + exclude) indices
        3. Sort only those indices by score.

        Parameters
        ----------
        exclude : extra words to remove from results (in addition to *word*).
        """
        if word not in self:
            raise KeyError(f"'{word}' not in vocabulary")

        query  = self.vec_norm(word)               # (D,)
        scores = self.normed @ query               # (V,)  cosine similarities

        # Words to exclude from the result
        excl_set = {word} | set(exclude or [])
        excl_idx = {self.word2idx[w] for w in excl_set if w in self}

        # Retrieve top_k + buffer then filter excluded
        buf   = top_k + len(excl_idx) + 1
        top_n_idx = np.argpartition(scores, -buf)[-buf:]
        top_n_idx = top_n_idx[np.argsort(scores[top_n_idx])[::-1]]

        results = []
        for idx in top_n_idx:
            if int(idx) in excl_idx:
                continue
            results.append((self.idx2word[idx], float(scores[idx])))
            if len(results) == top_k:
                break

        return results

    def analogy(
        self,
        a:    str,
        a_st: str,
        b:    str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Solve the analogy: *a* is to *a_st* as *b* is to ?

        Uses the 3CosAdd method (Mikolov et al. 2013 / Levy & Goldberg 2014):

            query = v_{a*} - v_a + v_b       (all L2-normalised before combining)
            b*    = argmax_{w \not\in {a,a*,b}} cos(v_w, query)

        The subtraction encodes "direction" in the embedding space —
        e.g. v_king - v_man isolates the "royalty" direction, and adding
        v_woman steers toward the female equivalent.

        Returns the top-*top_k* candidates and their cosine scores.
        """
        for word in (a, a_st, b):
            if word not in self:
                raise KeyError(f"'{word}' not in vocabulary")

        # Normalise each vector before combining to give equal weight.
        va, va_st, vb = (
            self.vec_norm(a),
            self.vec_norm(a_st),
            self.vec_norm(b),
        )

        query = va_st - va + vb                # (D,)

        # Re-normalise the combined query vector before cosine comparison.
        query_norm = query / (np.linalg.norm(query) + 1e-12)

        scores = self.normed @ query_norm       # (V,)

        excl_set = {a, a_st, b}
        excl_idx = {self.word2idx[w] for w in excl_set}

        buf = top_k + len(excl_idx) + 1
        top_idx = np.argpartition(scores, -buf)[-buf:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            if int(idx) in excl_idx:
                continue
            results.append((self.idx2word[idx], float(scores[idx])))
            if len(results) == top_k:
                break

        return results


def print_nearest(space: EmbeddingSpace, words: List[str], top_k: int = 8) -> None:
    """Print a nearest-neighbours table for each word in *words*."""
    print("\n" + "=" * 55)
    print("  Nearest Neighbours")
    print("=" * 55)
    for word in words:
        if word not in space:
            print(f"  '{word}' not in vocabulary — skipping")
            continue
        neighbours = space.most_similar(word, top_k=top_k)
        nstr = "  ".join(f"{w} ({s:.3f})" for w, s in neighbours)
        print(f"\n  {word!r:>12s} → {nstr}")
    print()


def print_analogies(
    space:    EmbeddingSpace,
    triples:  List[Tuple[str, str, str]],
) -> None:
    """
    Print analogy results.

    *triples* is a list of (a, a_star, b) tuples:
      a : a_star  ::  b : ???
    """
    print("\n" + "=" * 55)
    print("  Word Analogies  (a : a* :: b : ?)")
    print("=" * 55)
    for a, a_st, b in triples:
        try:
            results = space.analogy(a, a_st, b, top_k=3)
            pred    = results[0][0] if results else "N/A"
            others  = "  ".join(f"{w}({s:.3f})" for w, s in results[1:])
            print(f"  {a} : {a_st} :: {b} : {pred!r:12s}  |  {others}")
        except KeyError as e:
            print(f"  Skipping {(a, a_st, b)} — {e}")
    print()
