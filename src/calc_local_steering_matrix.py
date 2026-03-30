"""
calc_local_steering_matrix.py

Offline computation of LocalAlphaSteer parameters.

For each steered layer this script:
  1. Loads benign and harmful embeddings (same files as AlphaSteer).
  2. Clusters benign embeddings into K groups via K-means.
  3. Computes K local null-space projectors P_k.
  4. Solves for a shared Δ̃ via regularised pseudoinverse (same as AlphaSteer).
  5. Computes K local steering matrices M_k = P_k @ Δ̃.
  6. Saves per-layer centroids and local steering matrices to disk.

Saved artefact layout (a single .pt file):
    {
        "centroids":               tensor (num_layers, K, D),
        "local_steering_matrices": tensor (num_layers, K, D, D),
        "K":                       int,
        "temperature":             float,
    }

K=1 exactly recovers AlphaSteer (single global projector).

Usage example
-------------
python src/calc_local_steering_matrix.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --save_path data/steering_matrix/local_steering_llama3.1_K4.pt \
    --K 4 \
    --device cuda
"""

import argparse
import logging
import os
import pickle

import torch

torch.manual_seed(42)

from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.local_steering_utils import compute_local_steering_matrices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate LocalAlphaSteer matrices (K locally-gated projectors)"
    )
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Directory containing pre-extracted embeddings")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model nickname: llama3.1 | qwen2.5 | gemma2")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Path to save the local steering parameters (.pt)")
    parser.add_argument("--K", type=int, default=4,
                        help="Number of local projectors (K=1 == AlphaSteer)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for soft weights at inference")
    parser.add_argument("--lambda_reg", type=float, default=10.0,
                        help="Regularisation strength for Δ̃ estimation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    logger.info(f"model_name  : {args.model_name}")
    logger.info(f"K           : {args.K}")
    logger.info(f"temperature : {args.temperature}")
    logger.info(f"lambda_reg  : {args.lambda_reg}")

    embeds_dir = args.embedding_dir
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]

    # ------------------------------------------------------------------
    # Load benign embeddings (same files as AlphaSteer)
    # ------------------------------------------------------------------
    H_benign_train_10000 = torch.load(
        f"{embeds_dir}/embeds_benign_train.pt", map_location=device
    ).float()
    H_coconot_pref = torch.load(
        f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device
    ).float()
    H_coconot_original = torch.load(
        f"{embeds_dir}/embeds_coconot_original.pt", map_location=device
    ).float()

    indices_borderline = torch.randperm(H_coconot_original.size(0))[
        : 4000 - H_coconot_pref.size(0)
    ]
    H_benign_train = torch.cat(
        [H_benign_train_10000, H_coconot_original[indices_borderline], H_coconot_pref],
        dim=0,
    ).to(device)
    logger.info(f"H_benign_train shape: {H_benign_train.shape}")
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Load harmful embeddings (same files as AlphaSteer)
    # ------------------------------------------------------------------
    H_harmful_train_1000 = torch.load(
        f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location=device
    ).float()
    H_jailbreak_train_full = torch.load(
        f"{embeds_dir}/embeds_jailbreak_train.pt", map_location=device
    ).float()

    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)
    H_harmful_train_1000 = None
    H_jailbreak_train_full = None
    torch.cuda.empty_cache()
    logger.info(f"H_harmful_train shape: {H_harmful_train.shape}")

    # ------------------------------------------------------------------
    # Load refusal vectors
    # ------------------------------------------------------------------
    refusal_vectors_path = (
        f"data/refusal_vectors/RV/{args.model_name}_RV_refusal.pkl"
    )
    refusal_vectors = pickle.load(open(refusal_vectors_path, "rb"))
    refusal_vectors = torch.tensor(refusal_vectors, dtype=torch.float32).to(device)
    logger.info(f"refusal_vectors shape: {refusal_vectors.shape}")

    num_layers = refusal_vectors.shape[0]
    D = refusal_vectors.shape[1]
    K = args.K

    # Storage tensors (initialised to zero; only steered layers are filled)
    all_centroids = torch.zeros(num_layers, K, D, device=device)
    all_local_matrices = torch.zeros(num_layers, K, D, D, device=device)

    # ------------------------------------------------------------------
    # Per-layer computation
    # ------------------------------------------------------------------
    for layer, ratio in layers_ratio_list:
        logger.info(f"=== layer {layer}, null-space ratio {ratio}, K={K} ===")

        params = compute_local_steering_matrices(
            H_b=H_benign_train[:, layer, :],
            H_h=H_harmful_train[:, layer, :],
            refusal_vector=refusal_vectors[layer],
            K=K,
            abs_nullspace_ratio=ratio,
            lambda_reg=args.lambda_reg,
            temperature=args.temperature,
            device=str(device),
        )

        all_centroids[layer] = params.centroids
        all_local_matrices[layer] = params.local_steering_matrices

        logger.info(
            f"  centroids norm  : {params.centroids.norm():.4f}"
        )
        logger.info(
            f"  matrices norm   : {params.local_steering_matrices.norm():.4f}"
        )
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "centroids": all_centroids.cpu(),
            "local_steering_matrices": all_local_matrices.cpu(),
            "K": K,
            "temperature": args.temperature,
        },
        args.save_path,
    )
    logger.info(f"Saved LocalAlphaSteer params to {args.save_path}")

    # Cleanup
    H_benign_train = None
    H_harmful_train = None
    all_centroids = None
    all_local_matrices = None
    torch.cuda.empty_cache()
