import torch
import torch.nn.functional as F

from utils.steering_utils import null_space_l, null_space_projection_l

__all__ = [
    "cluster_benign_activations",
    "compute_local_projectors",
    "compute_kernel_weights",
    "compute_local_steering_matrices",
    "compute_composite_steering_matrix",
    "LocalProjectorFamily",
]


def cluster_benign_activations(H_b_layer, K, device="cuda:0", max_iter=100, seed=42):
    """
    Cluster benign activations at a single layer using K-means.

    Parameters:
    - H_b_layer: [N, d] - Benign activations at one layer
    - K: int - Number of clusters
    - device: str - Device for computation
    - max_iter: int - Maximum K-means iterations
    - seed: int - Random seed for reproducibility

    Returns:
    - centroids: [K, d] - Cluster centroids
    - assignments: [N] - Cluster assignment for each sample (long tensor)
    - bandwidths: [K] - Per-cluster bandwidth (average distance to centroid)
    """
    H_b_layer = H_b_layer.to(device).float()
    N, d = H_b_layer.shape

    if K == 1:
        centroid = H_b_layer.mean(dim=0, keepdim=True)
        assignments = torch.zeros(N, dtype=torch.long, device=device)
        bandwidth = torch.norm(H_b_layer - centroid, dim=1).mean().unsqueeze(0)
        return centroid, assignments, bandwidth

    # Initialize centroids via K-means++
    torch.manual_seed(seed)
    centroids = _kmeans_plus_plus_init(H_b_layer, K, device)

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        dists = torch.cdist(H_b_layer, centroids)  # [N, K]
        assignments = dists.argmin(dim=1)  # [N]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = assignments == k
            if mask.any():
                new_centroids[k] = H_b_layer[mask].mean(dim=0)
            else:
                # Reinitialize empty cluster to a random point
                new_centroids[k] = H_b_layer[torch.randint(N, (1,))]

        # Check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Compute per-cluster bandwidths
    bandwidths = torch.zeros(K, device=device)
    for k in range(K):
        mask = assignments == k
        if mask.any():
            bandwidths[k] = torch.norm(
                H_b_layer[mask] - centroids[k].unsqueeze(0), dim=1
            ).mean()
        else:
            bandwidths[k] = 1.0  # fallback

    return centroids, assignments, bandwidths


def _kmeans_plus_plus_init(X, K, device):
    """K-means++ initialization for centroids.

    Parameters:
    - X: [N, d] - Data points
    - K: int - Number of clusters
    - device: str - Computation device

    Returns:
    - centroids: [K, d]
    """
    N, d = X.shape
    centroids = torch.empty(K, d, device=device)

    # Pick first centroid uniformly at random
    idx = torch.randint(N, (1,), device=device).item()
    centroids[0] = X[idx]

    for k in range(1, K):
        dists = torch.cdist(X, centroids[:k])  # [N, k]
        min_dists = dists.min(dim=1).values  # [N]
        probs = min_dists ** 2
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()
        centroids[k] = X[idx]

    return centroids


def compute_local_projectors(
    H_b_layer, assignments, K, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0
):
    """
    Compute null-space projectors for each cluster of benign activations.

    Parameters:
    - H_b_layer: [N, d] - Benign activations at one layer
    - assignments: [N] - Cluster assignments
    - K: int - Number of clusters
    - min_null_space_ratio: float - Minimum null-space ratio for SVD
    - abs_nullspace_ratio: float - Absolute null-space ratio (overrides if > 0)

    Returns:
    - projectors: list of K tensors, each [d, d] - Per-cluster null-space projectors
    """
    d = H_b_layer.shape[1]
    device = H_b_layer.device
    projectors = []

    for k in range(K):
        mask = assignments == k
        if mask.sum() > 0:
            H_k = H_b_layer[mask]
            P_k = null_space_projection_l(
                H_k,
                min_null_space_ratio=min_null_space_ratio,
                abs_nullspace_ratio=abs_nullspace_ratio,
            )
            projectors.append(P_k)
        else:
            # Empty cluster: use identity as projector (no constraint)
            projectors.append(torch.eye(d, device=device))

    return projectors


def compute_kernel_weights(h, centroids, bandwidths):
    """
    Compute soft kernel gating weights for input activations.

    Parameters:
    - h: [B, d] or [d] - Input hidden states
    - centroids: [K, d] - Cluster centroids
    - bandwidths: [K] - Per-cluster bandwidths

    Returns:
    - weights: [B, K] or [K] - Normalized kernel weights (softmax over clusters)
    """
    single = h.dim() == 1
    if single:
        h = h.unsqueeze(0)

    # [B, K] distances
    diffs = h.unsqueeze(1) - centroids.unsqueeze(0)  # [B, K, d]
    sq_dists = (diffs ** 2).sum(dim=-1)  # [B, K]

    # Clamp bandwidths to avoid division by zero
    sigma_sq = (bandwidths ** 2).clamp(min=1e-8)  # [K]

    log_weights = -sq_dists / (2.0 * sigma_sq.unsqueeze(0))  # [B, K]
    weights = F.softmax(log_weights, dim=-1)  # [B, K]

    if single:
        weights = weights.squeeze(0)

    return weights


def compute_local_steering_matrices(
    H_h_layer,
    projectors,
    refusal_vector,
    centroids,
    bandwidths,
    lambda_reg=10.0,
    device="cuda:0",
):
    """
    Compute per-cluster steering solution vectors tilde_delta_k.

    For each cluster k, solve:
        H_{h,k} @ P_k @ tilde_delta_k ≈ refusal_vector
    weighted by kernel proximity to cluster k.

    Parameters:
    - H_h_layer: [N_h, d] - Harmful activations at one layer
    - projectors: list of K [d, d] tensors - Per-cluster null-space projectors
    - refusal_vector: [d] - Target refusal direction
    - centroids: [K, d] - Cluster centroids
    - bandwidths: [K] - Per-cluster bandwidths
    - lambda_reg: float - Regularization parameter
    - device: str

    Returns:
    - tilde_deltas: list of K [d, d] tensors - Per-cluster steering solutions
    - precomputed_S: list of K [d, d] tensors - P_k @ tilde_delta_k (for fast inference)
    """
    H_h_layer = H_h_layer.to(device).float()
    refusal_vector = refusal_vector.to(device).float()
    K = len(projectors)

    # Compute kernel weights for harmful samples
    weights = compute_kernel_weights(H_h_layer, centroids, bandwidths)  # [N_h, K]

    tilde_deltas = []
    precomputed_S = []

    for k in range(K):
        P_k = projectors[k].to(device)
        w_k = weights[:, k]  # [N_h]

        # Weight harmful activations by proximity to cluster k
        W_k = torch.diag(w_k.clamp(min=1e-8))  # [N_h, N_h]
        X = H_h_layer @ P_k  # [N_h, d]

        # Regularized weighted least-squares
        A = X.T @ W_k @ X + lambda_reg * (P_k.T @ P_k)  # [d, d]
        b = X.T @ (W_k @ refusal_vector.unsqueeze(0).expand(H_h_layer.shape[0], -1))  # [d, d]

        tilde_delta_k = torch.linalg.pinv(A) @ b
        tilde_deltas.append(tilde_delta_k)

        S_k = P_k @ tilde_delta_k
        precomputed_S.append(S_k)

        # Diagnostics
        result = X @ tilde_delta_k  # [N_h, d]
        avg_err = torch.norm(
            result - refusal_vector.unsqueeze(0), dim=1
        ).mean()
        print(
            f"  cluster {k}: avg_reconstruction_error={avg_err:.6f}, "
            f"refusal_norm={torch.norm(refusal_vector):.4f}"
        )

    return tilde_deltas, precomputed_S


def compute_composite_steering_matrix(h, centroids, bandwidths, precomputed_S):
    """
    Compute the position-dependent composite steering matrix M(h).

    M(h) = sum_k w_k(h) * S_k  where S_k = P_k @ tilde_delta_k

    Parameters:
    - h: [B, d] or [d] - Input hidden states
    - centroids: [K, d] - Cluster centroids
    - bandwidths: [K] - Per-cluster bandwidths
    - precomputed_S: list of K [d, d] tensors

    Returns:
    - M: [B, d, d] or [d, d] - Composite steering matrix
    """
    single = h.dim() == 1
    if single:
        h = h.unsqueeze(0)

    weights = compute_kernel_weights(h, centroids, bandwidths)  # [B, K]
    K = len(precomputed_S)

    # Stack precomputed matrices: [K, d, d]
    S_stack = torch.stack(precomputed_S, dim=0)

    # Weighted combination: [B, d, d]
    # weights: [B, K] -> [B, K, 1, 1]
    M = torch.einsum("bk,kij->bij", weights, S_stack)

    if single:
        M = M.squeeze(0)

    return M


class LocalProjectorFamily:
    """
    Encapsulates a family of K local null-space projectors with kernel gating.

    This is the core data structure for LocalAlphaSteer. When K=1, it degenerates
    to AlphaSteer's global projector.

    Attributes:
        K: Number of clusters
        centroids: [K, d] cluster centroids
        bandwidths: [K] per-cluster bandwidths
        projectors: list of K [d, d] null-space projectors
        tilde_deltas: list of K [d, d] steering solutions
        precomputed_S: list of K [d, d] precomputed P_k @ tilde_delta_k
    """

    def __init__(self, K, centroids, bandwidths, projectors, tilde_deltas, precomputed_S):
        self.K = K
        self.centroids = centroids
        self.bandwidths = bandwidths
        self.projectors = projectors
        self.tilde_deltas = tilde_deltas
        self.precomputed_S = precomputed_S

    def get_steering_matrix(self, h):
        """
        Compute M(h) for the given hidden states.

        Parameters:
        - h: [B, d] or [d]

        Returns:
        - M: [B, d, d] or [d, d]
        """
        return compute_composite_steering_matrix(
            h, self.centroids, self.bandwidths, self.precomputed_S
        )

    def get_kernel_weights(self, h):
        """Get kernel gating weights for diagnostics."""
        return compute_kernel_weights(h, self.centroids, self.bandwidths)

    def to(self, device):
        """Move all tensors to the specified device."""
        self.centroids = self.centroids.to(device)
        self.bandwidths = self.bandwidths.to(device)
        self.projectors = [p.to(device) for p in self.projectors]
        self.tilde_deltas = [d.to(device) for d in self.tilde_deltas]
        self.precomputed_S = [s.to(device) for s in self.precomputed_S]
        return self

    def state_dict(self):
        """Serialize to a dictionary for saving."""
        return {
            "K": self.K,
            "centroids": self.centroids,
            "bandwidths": self.bandwidths,
            "projectors": self.projectors,
            "tilde_deltas": self.tilde_deltas,
            "precomputed_S": self.precomputed_S,
        }

    @classmethod
    def from_state_dict(cls, state):
        """Deserialize from a saved dictionary."""
        return cls(
            K=state["K"],
            centroids=state["centroids"],
            bandwidths=state["bandwidths"],
            projectors=state["projectors"],
            tilde_deltas=state["tilde_deltas"],
            precomputed_S=state["precomputed_S"],
        )

    @classmethod
    def build(
        cls,
        H_b_layer,
        H_h_layer,
        refusal_vector,
        K=4,
        min_null_space_ratio=0.1,
        abs_nullspace_ratio=0.0,
        lambda_reg=10.0,
        device="cuda:0",
    ):
        """
        Build a LocalProjectorFamily from benign and harmful activations.

        This is the main entry point for constructing local projectors.
        When K=1, the result is equivalent to AlphaSteer's global projector.

        Parameters:
        - H_b_layer: [N_b, d] - Benign activations
        - H_h_layer: [N_h, d] - Harmful activations
        - refusal_vector: [d] - Refusal direction
        - K: int - Number of clusters
        - min_null_space_ratio: float
        - abs_nullspace_ratio: float
        - lambda_reg: float
        - device: str

        Returns:
        - family: LocalProjectorFamily instance
        """
        print(f"Building LocalProjectorFamily with K={K}")

        # Step 1: Cluster benign activations
        centroids, assignments, bandwidths = cluster_benign_activations(
            H_b_layer, K, device=device
        )
        print(f"  Cluster sizes: {[(assignments == k).sum().item() for k in range(K)]}")

        # Step 2: Compute per-cluster null-space projectors
        projectors = compute_local_projectors(
            H_b_layer.to(device),
            assignments,
            K,
            min_null_space_ratio=min_null_space_ratio,
            abs_nullspace_ratio=abs_nullspace_ratio,
        )

        # Step 3: Compute per-cluster steering solutions
        tilde_deltas, precomputed_S = compute_local_steering_matrices(
            H_h_layer,
            projectors,
            refusal_vector,
            centroids,
            bandwidths,
            lambda_reg=lambda_reg,
            device=device,
        )

        return cls(
            K=K,
            centroids=centroids,
            bandwidths=bandwidths,
            projectors=projectors,
            tilde_deltas=tilde_deltas,
            precomputed_S=precomputed_S,
        )
