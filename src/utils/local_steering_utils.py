"""
local_steering_utils.py — Utility functions for LocalAlphaSteer.

LocalAlphaSteer upgrades AlphaSteer's single global null-space projector to a
soft mixture of K locally-gated projectors.  K=1 recovers AlphaSteer exactly.

Offline computation (training-time)
------------------------------------
For each steered layer ℓ:
  1. Cluster benign embeddings H_b into K groups via K-means.
  2. For each cluster k, compute the local null-space projector P_k.
  3. Solve for a shared delta matrix Δ̃ (same regularised pseudoinverse as
     AlphaSteer, but using the mean projector P̄).
  4. Compute K local steering matrices  M_k = P_k @ Δ̃.
  5. Save centroids c_k  (K × D)  and local steering matrices M_k  (K × D × D).

Online computation (inference-time)
-------------------------------------
  w_k(h) = softmax( cosine_sim(h, c_k) / τ )
  M(h)   = Σ_k  w_k(h) · M_k
  h'     = h + λ · (h @ M(h))
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from .steering_utils import (
    null_space_l,
    null_space_projection_l,
    cal_tilde_delta_with_regularization_l,
    cal_steering_matrix_l,
)

__all__ = [
    "cluster_benign_embeddings",
    "local_null_space_projectors",
    "compute_local_steering_matrices",
    "compute_local_steering_vector",
    "LocalSteeringParams",
]


# ---------------------------------------------------------------------------
# K-means helpers
# ---------------------------------------------------------------------------

def _kmeans(X: torch.Tensor, K: int, n_iter: int = 100, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple Lloyd's K-means on CPU/GPU tensors.

    Parameters
    ----------
    X : (N, D) float tensor
    K : number of clusters
    n_iter : Lloyd iterations
    seed : random seed for initialisation

    Returns
    -------
    centroids  : (K, D)
    assignments: (N,) long tensor – cluster index per sample
    """
    torch.manual_seed(seed)
    N, D = X.shape
    if K > N:
        logger.warning(
            f"Requested K={K} clusters but only N={N} samples are available. "
            f"Clamping K to {N}."
        )
        K = N  # guard: cannot have more clusters than samples

    # K-means++ initialisation
    idx = [torch.randint(N, (1,), device=X.device).item()]
    for _ in range(1, K):
        dists = torch.cdist(X, X[idx], p=2).min(dim=1).values  # (N,)
        probs = (dists ** 2) / (dists ** 2).sum()
        idx.append(torch.multinomial(probs, 1).item())

    centroids = X[idx].clone()  # (K, D)

    for _ in range(n_iter):
        # Assignment step
        dists = torch.cdist(X, centroids, p=2)   # (N, K)
        assignments = dists.argmin(dim=1)          # (N,)

        # Update step
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=X.device)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = X[mask].mean(dim=0)
                counts[k] = mask.sum()
            else:
                # Keep old centroid if cluster is empty
                new_centroids[k] = centroids[k]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids, assignments


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def cluster_benign_embeddings(
    H_b: torch.Tensor,
    K: int,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster benign embeddings into K groups via K-means.

    Parameters
    ----------
    H_b : (N, D) – benign last-token hidden states at one layer
    K   : number of clusters
    n_iter : K-means iterations
    seed   : random seed

    Returns
    -------
    centroids   : (K, D)
    assignments : (N,) long – cluster index per sample
    """
    H_b = H_b.float()
    return _kmeans(H_b, K, n_iter=n_iter, seed=seed)


def local_null_space_projectors(
    H_b: torch.Tensor,
    assignments: torch.Tensor,
    K: int,
    abs_nullspace_ratio: float = 0.0,
    min_null_space_ratio: float = 0.1,
    min_cluster_size: int = 10,
) -> list[torch.Tensor]:
    """
    Compute a per-cluster null-space projector P_k for each of the K clusters.

    For clusters with fewer than *min_cluster_size* samples, the null-space is
    computed on the full benign set as a fallback (mirrors AlphaSteer behaviour).

    Parameters
    ----------
    H_b               : (N, D)
    assignments       : (N,) long – output of cluster_benign_embeddings
    K                 : number of clusters
    abs_nullspace_ratio : forwarded to null_space_projection_l
    min_null_space_ratio: forwarded to null_space_projection_l
    min_cluster_size  : minimum cluster size; use global H_b for smaller clusters

    Returns
    -------
    projectors : list of K tensors, each (D, D)
    """
    H_b = H_b.float()
    projectors = []
    for k in range(K):
        mask = assignments == k
        H_k = H_b[mask]
        if H_k.shape[0] < min_cluster_size:
            # Cluster k has fewer than min_cluster_size samples.
            # Fall back to the full benign set so that the projector is still
            # well-defined.  Note: small clusters that fall back share the
            # same global projector, which reduces but does not eliminate the
            # local constraint benefit.  Increase min_cluster_size or reduce K
            # if many clusters hit this path.
            logger.warning(
                f"Cluster {k} has only {H_k.shape[0]} samples "
                f"(< min_cluster_size={min_cluster_size}). "
                "Falling back to the global benign set for this projector."
            )
            H_k = H_b
        P_k = null_space_projection_l(
            H_k,
            min_null_space_ratio=min_null_space_ratio,
            abs_nullspace_ratio=abs_nullspace_ratio,
        )
        projectors.append(P_k)
    return projectors


def compute_local_steering_matrices(
    H_b: torch.Tensor,
    H_h: torch.Tensor,
    refusal_vector: torch.Tensor,
    K: int,
    abs_nullspace_ratio: float = 0.0,
    min_null_space_ratio: float = 0.1,
    lambda_reg: float = 10.0,
    temperature: float = 1.0,
    device: str = "cuda",
) -> "LocalSteeringParams":
    """
    Full offline computation of LocalAlphaSteer parameters for one layer.

    Steps
    -----
    1. K-means cluster H_b → centroids, assignments
    2. Per-cluster null-space projectors P_k
    3. Mean projector P̄ used for Δ̃ estimation
    4. Δ̃ via regularised pseudoinverse (same as AlphaSteer)
    5. M_k = P_k @ Δ̃  for each k

    Parameters
    ----------
    H_b              : (N_b, D) benign embeddings at this layer
    H_h              : (N_h, D) harmful embeddings at this layer
    refusal_vector   : (D,) target refusal direction
    K                : number of local projectors
    abs_nullspace_ratio, min_null_space_ratio : null-space params
    lambda_reg       : regularisation strength
    temperature      : soft-weight temperature τ (stored, not used offline)
    device           : torch device string

    Returns
    -------
    LocalSteeringParams dataclass with centroids and local_steering_matrices
    """
    H_b = H_b.float().to(device)
    H_h = H_h.float().to(device)
    refusal_vector = refusal_vector.float().to(device)

    # Step 1 — cluster benign embeddings
    centroids, assignments = cluster_benign_embeddings(H_b, K)
    centroids = centroids.to(device)

    # Step 2 — per-cluster null-space projectors
    projectors = local_null_space_projectors(
        H_b, assignments, K,
        abs_nullspace_ratio=abs_nullspace_ratio,
        min_null_space_ratio=min_null_space_ratio,
    )

    # Step 3 — mean projector for Δ̃ estimation
    P_mean = torch.stack(projectors, dim=0).mean(dim=0)  # (D, D)

    # Step 4 — Δ̃ via regularised pseudoinverse (same formula as AlphaSteer)
    tilde_delta = cal_tilde_delta_with_regularization_l(
        H_h, P_mean, refusal_vector, lambda_reg=lambda_reg, device=device
    )

    # Step 5 — local steering matrices M_k = P_k @ Δ̃
    local_matrices = []
    for P_k in projectors:
        M_k = cal_steering_matrix_l(P_k.to(device), tilde_delta, device=device)
        local_matrices.append(M_k)

    local_steering_matrices = torch.stack(local_matrices, dim=0)  # (K, D, D)

    return LocalSteeringParams(
        centroids=centroids,
        local_steering_matrices=local_steering_matrices,
        temperature=temperature,
    )


def compute_local_steering_vector(
    h: torch.Tensor,
    centroids: torch.Tensor,
    local_steering_matrices: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Online computation of the LocalAlphaSteer steering vector for a batch of
    hidden states.

    Parameters
    ----------
    h                       : (B, D) last-token hidden states
    centroids               : (K, D) cluster centroids
    local_steering_matrices : (K, D, D) local steering matrices M_k
    temperature             : softmax temperature τ

    Returns
    -------
    steering_vector : (B, D)
    """
    h = h.float()
    centroids = centroids.float().to(h.device)
    local_steering_matrices = local_steering_matrices.float().to(h.device)

    # Cosine similarity: (B, K)
    h_norm = torch.nn.functional.normalize(h, dim=-1)                     # (B, D)
    c_norm = torch.nn.functional.normalize(centroids, dim=-1)             # (K, D)
    sim = h_norm @ c_norm.T                                                 # (B, K)

    # Soft weights: (B, K)
    weights = torch.softmax(sim / temperature, dim=-1)

    # Weighted sum of local steering matrices: M(h) = Σ_k w_k(h) M_k
    # local_steering_matrices : (K, D, D)
    # weights                 : (B, K)
    # result                  : (B, D, D)
    M_h = torch.einsum("bk,kij->bij", weights, local_steering_matrices)   # (B, D, D)

    # Steering vector: v = h @ M(h), shape (B, D)
    steering_vector = torch.bmm(h.unsqueeze(1), M_h).squeeze(1)           # (B, D)

    return steering_vector


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class LocalSteeringParams:
    """
    Container for per-layer LocalAlphaSteer parameters.

    Attributes
    ----------
    centroids               : (K, D) cluster centroids
    local_steering_matrices : (K, D, D) local steering matrices M_k
    temperature             : softmax temperature τ
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        local_steering_matrices: torch.Tensor,
        temperature: float = 1.0,
    ):
        self.centroids = centroids
        self.local_steering_matrices = local_steering_matrices
        self.temperature = temperature

    @property
    def K(self) -> int:
        return self.centroids.shape[0]

    @property
    def D(self) -> int:
        return self.centroids.shape[1]

    def to(self, device) -> "LocalSteeringParams":
        self.centroids = self.centroids.to(device)
        self.local_steering_matrices = self.local_steering_matrices.to(device)
        return self

    def state_dict(self) -> dict:
        return {
            "centroids": self.centroids,
            "local_steering_matrices": self.local_steering_matrices,
            "temperature": self.temperature,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "LocalSteeringParams":
        return cls(
            centroids=d["centroids"],
            local_steering_matrices=d["local_steering_matrices"],
            temperature=float(d["temperature"]),
        )
