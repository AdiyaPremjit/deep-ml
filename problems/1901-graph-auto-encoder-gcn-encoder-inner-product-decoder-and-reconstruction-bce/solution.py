import numpy as np


def _norm_adj(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    A_tilde = A + np.eye(N, dtype=float)
    deg = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def gae_encode(A: np.ndarray, X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    A_hat = _norm_adj(A)
    hidden = np.maximum(0.0, A_hat @ X @ W1)
    return A_hat @ hidden @ W2


def gae_logits(Z: np.ndarray, pairs) -> np.ndarray:
    if len(pairs) == 0:
        return np.array([], dtype=float)
    idx = np.asarray(pairs, dtype=int)
    return np.sum(Z[idx[:, 0]] * Z[idx[:, 1]], axis=1)


def gae_loss(A: np.ndarray, Z: np.ndarray, pos_weight=None) -> float:
    A = np.asarray(A, dtype=float)
    N = A.shape[0]
    logits = Z @ Z.T
    pos_sum = np.sum(A)
    if pos_weight is None:
        pos_weight = (N * N - pos_sum) / pos_sum
    loss = -pos_weight * A * _log_sigmoid(logits) - (1.0 - A) * _log_sigmoid(-logits)
    return float(np.mean(loss))


def sampled_gae_loss(Z: np.ndarray, pos_pairs, neg_pairs) -> float:
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)
    pos_bce = -_log_sigmoid(pos_logits)
    neg_bce = -_log_sigmoid(-neg_logits)
    return float(np.mean(np.concatenate([pos_bce, neg_bce])))


def vgae_sample(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return mu + np.exp(0.5 * logvar) * eps


def vgae_kl(mu: np.ndarray, logvar: np.ndarray) -> float:
    kl_per_node = -0.5 * np.sum(1.0 + logvar - mu**2 - np.exp(logvar), axis=1)
    return float(np.mean(kl_per_node))
