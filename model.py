import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Element-wise sigmoid.
    """
    return 1 / (1.0 + np.exp(-x))


class Word2Vec:
    """
    Skip-gram Word2Vec trained with Negative Sampling (SGNS).

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    embed_dim : int
        Dimensionality D of the embedding vectors.
    seed : int
        Random seed for reproducible initialisation.
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        # --- Initialisation ---------------------------------------------------
        # W_in is initialised with small uniform noise so symmetry is broken
        # but activations start near zero (common practice for word2vec).
        scale = 0.5 / embed_dim
        self.W_in  = rng.uniform(-scale, scale, (vocab_size, embed_dim))  # (V, D)

        # W_out is initialised to zero — the model still works because W_in
        # is already asymmetric.
        self.W_out = np.zeros((vocab_size, embed_dim))                    # (V, D)

        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim

    def forward(
        self,
        center_idx:  int,
        context_idx: int,
        neg_indices: np.ndarray,   # shape (K,)
    ) -> tuple:
        """
        Compute SGNS loss and all gradients for one (center, context, negatives)
        training example.

        Returns
        -------
        loss      : float           -L_NEG  (scalar, ≥ 0)
        grad_v_c  : ndarray (D,)    d loss / d (center embedding v_c)
        grad_u_p  : ndarray (D,)    d loss / d (positive context embedding u_p)
        grad_U_neg: ndarray (K, D)  d loss / d (negative sample embeddings u_k)
        """
        # ── Fetch embeddings ──────────────────────────────────────────────
        v_c   = self.W_in[center_idx]    # center embedding        (D,)
        u_p   = self.W_out[context_idx]  # positive context embed  (D,)
        U_neg = self.W_out[neg_indices]  # negative sample embeds  (K, D)

        # ── Positive pair ─────────────────────────────────────────────────
        # s_o = v_c · u_p
        similarity_p     = np.dot(v_c, u_p)
        sig_pos = sigmoid(similarity_p)

        # ── Negative pairs ────────────────────────────────────────────────
        # s_k = v_c · u_k  for all k — shape (K,)
        similarity_neg = U_neg @ v_c                   # (K,)
        sig_neg = sigmoid(-similarity_neg)

        # ── Loss = −L_NEG ────────────────────────────────────────────────
        _eps  = 1e-10
        loss  = -(np.log(sig_pos + _eps) + np.sum(np.log(sig_neg + _eps)))

        # ── Gradients ────────────────────────────────────────────────────
        err_pos = sig_pos - 1.0
        err_neg = 1.0 - sig_neg

        # d loss / d v_c
        grad_v_c   = err_pos * u_p + err_neg @ U_neg   # (D,)

        # d loss / d u_p  — eq. (2)
        grad_u_p   = err_pos * v_c                     # (D,)

        # d loss / d u_k  — eq. (3), one row per negative sample
        grad_U_neg = np.outer(err_neg, v_c)            # (K, D)

        return loss, grad_v_c, grad_u_p, grad_U_neg

    def update(
        self,
        center_idx:  int,
        context_idx: int,
        neg_indices: np.ndarray,
        grad_v_c:    np.ndarray,
        grad_u_p:    np.ndarray,
        grad_U_neg:  np.ndarray,
        lr:          float,
    ) -> None:
        """
        Update parameters with SGD.
        
        np.add.at is used for neg_indices because the same word index may
        appear more than once in neg_indices (though rare); using fancy
        indexing  W_out[neg_indices] -= ...  would silently drop duplicate
        gradient contributions, whereas add.at accumulates them correctly.
        """
        self.W_in[center_idx]   -= lr * grad_v_c
        self.W_out[context_idx] -= lr * grad_u_p
        np.add.at(self.W_out, neg_indices, -lr * grad_U_neg)


    def train_pair(
        self,
        center_idx:  int,
        context_idx: int,
        neg_indices: np.ndarray,
        lr:          float,
    ) -> float:
        """Forward pass, gradient computation, and SGD update in one call."""
        loss, grad_v_c, grad_u_p, grad_U_neg = self.forward(
            center_idx, context_idx, neg_indices
        )
        self.update(center_idx, context_idx, neg_indices,
                    grad_v_c, grad_u_p, grad_U_neg, lr)
        return loss

    @property
    def vectors(self) -> np.ndarray:
        """
        Final word vectors — the input (center) embeddings W_in.
        """
        return self.W_in
