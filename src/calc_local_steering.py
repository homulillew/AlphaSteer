"""
Calculate locally-gated steering matrices for LocalAlphaSteer.

This script extends AlphaSteer's steering matrix calculation by:
1. Clustering benign activations into K local neighborhoods
2. Computing local null-space projectors per neighborhood
3. Computing local steering directions per neighborhood
4. Saving all components (anchors, local matrices, temperature) for inference

Usage:
    python src/calc_local_steering.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --save_path data/steering_matrix/local_steering_llama3.1.pt \
        --num_anchors 8 \
        --device cuda
"""

import torch
torch.manual_seed(42)
from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.local_steering_utils import (
    compute_anchors,
    auto_calibrate_temperature,
    compute_gating_weights,
    compute_local_null_space_projector,
    compute_local_steering_direction,
)

import pickle
import os
import argparse

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate locally-gated steering matrices for LocalAlphaSteer"
    )
    parser.add_argument(
        "--embedding_dir", type=str, required=True,
        help="Directory containing embeddings"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Model name (e.g., llama3.1, qwen2.5, gemma2)"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to save the local steering parameters"
    )
    parser.add_argument(
        "--num_anchors", type=int, default=8,
        help="Number of local anchors K (default: 8)"
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Temperature parameter (auto-calibrated if not provided)"
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=10.0,
        help="Regularization parameter for steering direction (default: 10.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    K = args.num_anchors

    logger.info(f"model_name: {args.model_name}")
    logger.info(f"num_anchors K: {K}")

    embeds_dir = args.embedding_dir
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]

    # Load benign embeddings (same as AlphaSteer)
    H_benign_train_10000 = torch.load(
        f"{embeds_dir}/embeds_benign_train.pt", map_location=device
    ).float()
    H_coconot_pref = torch.load(
        f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device
    ).float()
    H_coconot_original = torch.load(
        f"{embeds_dir}/embeds_coconot_original.pt", map_location=device
    ).float()

    indices_borderline = torch.randperm(
        H_coconot_original.size(0)
    )[:4000 - H_coconot_pref.size(0)]
    H_benign_train = torch.cat([
        H_benign_train_10000,
        H_coconot_original[indices_borderline],
        H_coconot_pref
    ], dim=0).to(device)

    logger.info(f"H_benign_train shape: {H_benign_train.shape}")
    torch.cuda.empty_cache()

    # Load harmful embeddings (same as AlphaSteer)
    H_harmful_train_1000 = torch.load(
        f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location=device
    ).float()
    H_jailbreak_train_full = torch.load(
        f"{embeds_dir}/embeds_jailbreak_train.pt", map_location=device
    ).float()

    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    H_harmful_train = torch.cat(
        [H_harmful_train_1000, H_jailbreak_train], dim=0
    )
    H_harmful_train_1000 = None
    H_jailbreak_train_full = None
    torch.cuda.empty_cache()
    logger.info(f"H_harmful_train shape: {H_harmful_train.shape}")

    # Load refusal vectors
    refusal_vectors_path = (
        f"data/refusal_vectors/RV/{args.model_name}_RV_refusal.pkl"
    )
    refusal_vectors = pickle.load(open(refusal_vectors_path, "rb"))
    refusal_vectors = torch.tensor(
        refusal_vectors, dtype=torch.float32
    ).to(device)
    logger.info(f"refusal vectors shape: {refusal_vectors.shape}")

    num_layer = refusal_vectors.shape[0]
    d_model = refusal_vectors.shape[1]

    # Storage for all layers
    local_steering_params = {}

    for layer, ratio in layers_ratio_list:
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer {layer}, null-space ratio: {ratio}")
        logger.info(f"{'='*60}")

        H_b_layer = H_benign_train[:, layer, :]   # [N_b, d]
        H_m_layer = H_harmful_train[:, layer, :]   # [N_m, d]
        r_layer = refusal_vectors[layer]            # [d]

        # Step 1: Compute anchors
        anchors, assignments = compute_anchors(H_b_layer, K, device=device)
        logger.info(f"Computed {K} anchors for layer {layer}")

        # Log cluster sizes
        for k_idx in range(K):
            count = (assignments == k_idx).sum().item()
            logger.info(f"  Anchor {k_idx}: {count} benign samples assigned")

        # Step 2: Auto-calibrate temperature
        tau = args.tau if args.tau is not None else auto_calibrate_temperature(anchors)

        # Step 3: Compute gating weights for all benign and harmful samples
        w_b = compute_gating_weights(H_b_layer, anchors, tau)  # [N_b, K]
        w_m = compute_gating_weights(H_m_layer, anchors, tau)  # [N_m, K]

        # Step 4: Compute local projectors and steering directions
        S_list = []
        diagnostics = {'mu_k': [], 'S_norm': []}

        for k_idx in range(K):
            logger.info(f"  Anchor {k_idx}: computing local projector")

            # Local null-space projector
            Q_k, P_k = compute_local_null_space_projector(
                H_b_layer, w_b[:, k_idx],
                abs_nullspace_ratio=ratio,
                device=device,
            )

            # Diagnostic: refusal direction compatibility
            P_r = P_k @ r_layer
            mu_k = torch.norm(P_r) / torch.norm(r_layer)
            diagnostics['mu_k'].append(mu_k.item())
            logger.info(
                f"  Anchor {k_idx}: mu_k (refusal compatibility) = {mu_k:.4f}"
            )

            # Local steering direction
            delta_k = compute_local_steering_direction(
                H_m_layer, P_k, r_layer,
                weights_k_m=w_m[:, k_idx],
                lambda_reg=args.lambda_reg,
                device=device,
            )

            # Local steering matrix
            S_k = P_k @ delta_k
            S_list.append(S_k)

            S_norm = torch.norm(S_k).item()
            diagnostics['S_norm'].append(S_norm)
            logger.info(f"  Anchor {k_idx}: S_k norm = {S_norm:.6f}")

        # Stack local steering matrices
        S_stack = torch.stack(S_list, dim=0)  # [K, d, d]

        # Store parameters for this layer
        local_steering_params[layer] = {
            'anchors': anchors.cpu(),           # [K, d]
            'steering_matrices': S_stack.cpu(), # [K, d, d]
            'tau': tau,
            'diagnostics': diagnostics,
        }

        # Summary diagnostics
        logger.info(f"Layer {layer} summary:")
        logger.info(f"  Mean mu_k: {sum(diagnostics['mu_k'])/K:.4f}")
        logger.info(f"  Mean S_norm: {sum(diagnostics['S_norm'])/K:.6f}")

    # Save all local steering parameters
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(local_steering_params, args.save_path)
    logger.info(f"\nLocal steering parameters saved to {args.save_path}")
    logger.info(f"Contains parameters for {len(local_steering_params)} layers")
    logger.info(f"Each layer has {K} anchors")

    # Cleanup
    H_benign_train = None
    H_harmful_train = None
    torch.cuda.empty_cache()
