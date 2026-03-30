import torch
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

__all__ = [
    "compute_anchors",
    "compute_gating_weights",
    "compute_local_null_space_projector",
    "compute_local_steering_direction",
    "compute_local_steering_matrices",
    "compute_blended_steering",
    "auto_calibrate_temperature",
]


def compute_anchors(H_b_layer, K, device="cuda:0"):
    """
    Compute K anchor points from benign activations using k-means clustering.

    Parameters:
    - H_b_layer: [N_b, d] - Benign activations at a single layer
    - K: int - Number of anchors
    - device: Device for computation

    Returns:
    - anchors: [K, d] - Anchor centers
    - assignments: [N_b] - Cluster assignments for each benign sample
    """
    H_b_layer = H_b_layer.to(device).float()
    N_b, d = H_b_layer.shape

    if K >= N_b:
        logger.warning(f"K={K} >= N_b={N_b}, using all points as anchors")
        return H_b_layer.clone(), torch.arange(N_b, device=device)

    # Initialize with k-means++ style initialization
    indices = [torch.randint(0, N_b, (1,), device=device).item()]
    for _ in range(1, K):
        dists = torch.cdist(
            H_b_layer, H_b_layer[indices], p=2
        ).min(dim=1).values
        probs = dists ** 2
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()
        indices.append(idx)

    anchors = H_b_layer[indices].clone()

    # Run k-means iterations
    max_iter = 100
    for iteration in range(max_iter):
        # Assign each point to nearest anchor
        dists = torch.cdist(H_b_layer, anchors, p=2)  # [N_b, K]
        assignments = dists.argmin(dim=1)  # [N_b]

        # Update anchors
        new_anchors = torch.zeros_like(anchors)
        for k in range(K):
            mask = assignments == k
            if mask.sum() > 0:
                new_anchors[k] = H_b_layer[mask].mean(dim=0)
            else:
                # Reinitialize empty clusters
                new_anchors[k] = H_b_layer[torch.randint(0, N_b, (1,))].squeeze(0)

        # Check convergence
        shift = (new_anchors - anchors).norm(dim=1).max().item()
        anchors = new_anchors
        if shift < 1e-6:
            logger.info(f"K-means converged at iteration {iteration + 1}")
            break

    # Final assignment
    dists = torch.cdist(H_b_layer, anchors, p=2)
    assignments = dists.argmin(dim=1)

    return anchors, assignments


def auto_calibrate_temperature(anchors):
    """
    Auto-calibrate the temperature parameter tau from inter-anchor distances.

    Uses the median squared inter-anchor distance as tau.

    Parameters:
    - anchors: [K, d] - Anchor centers

    Returns:
    - tau: float - Temperature parameter
    """
    K = anchors.shape[0]
    if K <= 1:
        return 1.0

    dists = torch.cdist(anchors, anchors, p=2)  # [K, K]
    # Get upper triangle (exclude diagonal)
    mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=anchors.device), diagonal=1)
    pairwise_dists = dists[mask]

    tau = (pairwise_dists ** 2).median().item()
    if tau < 1e-10:
        tau = 1.0
        logger.warning("Inter-anchor distances too small, using tau=1.0")

    logger.info(f"Auto-calibrated temperature tau={tau:.4f}")
    return tau


def compute_gating_weights(h, anchors, tau):
    """
    Compute soft gating weights for activation h relative to anchors.

    w_k(h) = exp(-||h - z_k||^2 / tau) / sum_j exp(-||h - z_j||^2 / tau)

    Parameters:
    - h: [B, d] or [d] - Activation(s) to gate
    - anchors: [K, d] - Anchor centers
    - tau: float - Temperature parameter

    Returns:
    - weights: [B, K] or [K] - Soft gating weights
    """
    squeeze = False
    if h.dim() == 1:
        h = h.unsqueeze(0)
        squeeze = True

    # [B, K]
    sq_dists = torch.cdist(h, anchors, p=2) ** 2
    logits = -sq_dists / tau
    weights = F.softmax(logits, dim=-1)

    if squeeze:
        weights = weights.squeeze(0)

    return weights


def compute_local_null_space_projector(H_b_layer, weights_k, abs_nullspace_ratio=0.3,
                                       min_effective_samples=10, device="cuda:0"):
    """
    Compute the local null-space projector for a single anchor neighborhood.

    Uses soft-weighted SVD: weight each benign sample by sqrt(w_k(h_b))
    before computing the null-space.

    Parameters:
    - H_b_layer: [N_b, d] - All benign activations at this layer
    - weights_k: [N_b] - Gating weights for anchor k (w_k(h_b^(i)))
    - abs_nullspace_ratio: Ratio of null-space dimension to total dimension
    - min_effective_samples: Minimum effective sample count for stable SVD
    - device: Device for computation

    Returns:
    - Q_k: [d, n_k] - Null-space basis vectors
    - P_k: [d, d] - Null-space projection matrix (Q_k @ Q_k^T)
    """
    H_b_layer = H_b_layer.to(device).float()
    weights_k = weights_k.to(device).float()

    d = H_b_layer.shape[1]

    # Effective sample count
    eff_n = (weights_k.sum() ** 2) / (weights_k ** 2).sum()
    if eff_n < min_effective_samples:
        logger.warning(
            f"Effective samples={eff_n:.1f} < {min_effective_samples}, "
            f"using uniform weights as fallback"
        )
        weights_k = torch.ones_like(weights_k) / len(weights_k)

    # Weight the benign activations: diag(sqrt(w)) @ H_b
    sqrt_weights = torch.sqrt(weights_k).unsqueeze(1)  # [N_b, 1]
    H_weighted = sqrt_weights * H_b_layer  # [N_b, d]

    # SVD of the weighted covariance
    _, S, Vh = torch.linalg.svd(H_weighted.T @ H_weighted)

    num_null = int(d * abs_nullspace_ratio)
    num_null = max(num_null, 1)

    # Null-space basis: last num_null right singular vectors
    Q_k = Vh[-num_null:, :].T.conj()  # [d, num_null]

    # Projection matrix
    P_k = Q_k @ Q_k.T  # [d, d]

    return Q_k, P_k


def compute_local_steering_direction(H_m_layer, P_k, refusal_vector,
                                      weights_k_m=None, lambda_reg=10.0,
                                      device="cuda:0"):
    """
    Compute the local steering direction for a single anchor.

    Solves: min_delta ||H_m P_k delta - r||^2 + lambda ||P_k delta||^2

    Parameters:
    - H_m_layer: [N_m, d] - Harmful activations at this layer
    - P_k: [d, d] - Local null-space projector
    - refusal_vector: [d] - Target refusal direction
    - weights_k_m: [N_m] or None - Optional gating weights for harmful samples
    - lambda_reg: float - Regularization parameter
    - device: Device for computation

    Returns:
    - delta_k: [d, d] - Local steering direction matrix
    """
    H_m_layer = H_m_layer.to(device).float()
    P_k = P_k.to(device).float()
    refusal_vector = refusal_vector.to(device).float()

    X = H_m_layer @ P_k  # [N_m, d]

    if weights_k_m is not None:
        weights_k_m = weights_k_m.to(device).float()
        sqrt_w = torch.sqrt(weights_k_m).unsqueeze(1)  # [N_m, 1]
        X_w = sqrt_w * X
        r_w = sqrt_w * refusal_vector.unsqueeze(0).expand_as(X)
    else:
        X_w = X
        r_w = refusal_vector.unsqueeze(0).expand(X.shape[0], -1)

    # Normal equation: (X^T X + lambda P^T P) delta = X^T r
    A = X_w.T @ X_w + lambda_reg * (P_k.T @ P_k)
    b = X_w.T @ r_w

    delta_k = torch.linalg.pinv(A) @ b  # [d, d]

    # Diagnostic: reconstruction error
    result = X @ delta_k
    avg_error = torch.norm(result - refusal_vector) / H_m_layer.shape[0]
    logger.info(f"Local steering: avg_reconstruction_error={avg_error:.6f}, "
                f"refusal_norm={torch.norm(refusal_vector):.6f}")

    return delta_k


def compute_local_steering_matrices(H_b_layer, H_m_layer, refusal_vector,
                                     K, abs_nullspace_ratio=0.3,
                                     lambda_reg=10.0, tau=None,
                                     device="cuda:0"):
    """
    Compute all local steering components for a single layer.

    This is the main entry point that computes anchors, local projectors,
    local steering directions, and temperature for one layer.

    Parameters:
    - H_b_layer: [N_b, d] - Benign activations
    - H_m_layer: [N_m, d] - Harmful activations
    - refusal_vector: [d] - Refusal direction
    - K: int - Number of anchors
    - abs_nullspace_ratio: Null-space ratio for local projectors
    - lambda_reg: Regularization for steering direction
    - tau: Temperature (auto-calibrated if None)
    - device: Device for computation

    Returns:
    - anchors: [K, d] - Anchor centers
    - Q_list: list of [d, n_k] tensors - Local null-space bases
    - S_list: list of [d, d] tensors - Local steering matrices (P_k @ delta_k)
    - tau: float - Temperature parameter
    """
    H_b_layer = H_b_layer.to(device).float()
    H_m_layer = H_m_layer.to(device).float()
    refusal_vector = refusal_vector.to(device).float()

    # Step 1: Compute anchors
    anchors, _ = compute_anchors(H_b_layer, K, device=device)
    logger.info(f"Computed {K} anchors")

    # Step 2: Auto-calibrate temperature if not provided
    if tau is None:
        tau = auto_calibrate_temperature(anchors)

    # Step 3: Compute gating weights for benign and harmful activations
    w_b = compute_gating_weights(H_b_layer, anchors, tau)  # [N_b, K]
    w_m = compute_gating_weights(H_m_layer, anchors, tau)  # [N_m, K]

    Q_list = []
    S_list = []

    for k in range(K):
        logger.info(f"  Anchor {k}: computing local projector and steering direction")

        # Step 4: Local null-space projector
        Q_k, P_k = compute_local_null_space_projector(
            H_b_layer, w_b[:, k],
            abs_nullspace_ratio=abs_nullspace_ratio,
            device=device
        )
        Q_list.append(Q_k)

        # Step 5: Local steering direction
        delta_k = compute_local_steering_direction(
            H_m_layer, P_k, refusal_vector,
            weights_k_m=w_m[:, k],
            lambda_reg=lambda_reg,
            device=device
        )

        # Step 6: Local steering matrix
        S_k = P_k @ delta_k  # [d, d]
        S_list.append(S_k)

        S_norm = torch.norm(S_k).item()
        logger.info(f"  Anchor {k}: S_k norm={S_norm:.6f}")

    return anchors, Q_list, S_list, tau


def compute_blended_steering(h, anchors, S_list, tau, strength=1.0):
    """
    Compute the position-dependent steering vector for activation h.

    s(h) = strength * (sum_k w_k(h) * S_k) @ h

    Parameters:
    - h: [B, d] or [d] - Input activation(s)
    - anchors: [K, d] - Anchor centers
    - S_list: list of [d, d] tensors - Local steering matrices
    - tau: float - Temperature
    - strength: float - Overall steering strength

    Returns:
    - steering_vector: [B, d] or [d] - Steering vector to add to h
    """
    squeeze = False
    if h.dim() == 1:
        h = h.unsqueeze(0)
        squeeze = True

    B, d = h.shape
    K = anchors.shape[0]
    device = h.device

    # Compute gating weights: [B, K]
    weights = compute_gating_weights(h, anchors, tau)

    # Blend steering matrices: [B, d, d]
    # Efficient: compute weighted sum of S_k for each sample
    S_stack = torch.stack(S_list, dim=0)  # [K, d, d]
    blended_S = torch.einsum('bk,kij->bij', weights, S_stack)  # [B, d, d]

    # Apply steering: s(h) = strength * blended_S @ h
    steering_vector = strength * torch.bmm(
        blended_S, h.unsqueeze(-1)
    ).squeeze(-1)  # [B, d]

    if squeeze:
        steering_vector = steering_vector.squeeze(0)

    return steering_vector
