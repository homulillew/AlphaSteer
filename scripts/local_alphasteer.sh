#!/bin/bash
# LocalAlphaSteer Pipeline: Locally-Gated Projector Steering
#
# This script runs the complete LocalAlphaSteer pipeline:
# 1. Extract embeddings (reuses AlphaSteer's extraction)
# 2. Calculate locally-gated steering matrices
# 3. Generate responses with local steering
#
# Usage: bash scripts/local_alphasteer.sh

set -e

# Configuration
TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda:0
NUM_ANCHORS=8  # Number of local anchors K

echo "========================================"
echo "LocalAlphaSteer Pipeline"
echo "Model: ${MODEL_NAME}"
echo "Num Anchors K: ${NUM_ANCHORS}"
echo "========================================"

# Stage 1: Extract embeddings (same as AlphaSteer)
echo ""
echo "Stage 1: Extracting embeddings..."
echo "========================================"

mkdir -p ${EMBEDDING_DIR}

declare -A PROMPT_COLUMNS
PROMPT_COLUMNS[benign_train]="query"
PROMPT_COLUMNS[harmful_train_1000]="query"
PROMPT_COLUMNS[jailbreak_train]="prompt"
PROMPT_COLUMNS[coconot_original]="prompt"
PROMPT_COLUMNS[coconot_pref]="prompt"
PROMPT_COLUMNS[benign_val]="query"
PROMPT_COLUMNS[harmful_val]="query"
PROMPT_COLUMNS[borderline_val]="prompt"

for file in ${TRAIN_VAL_DIR}/*.json; do
    filename=$(basename "$file" .json)
    output_file="${EMBEDDING_DIR}/embeds_${filename}.pt"

    if [ -f "$output_file" ]; then
        echo "Skipping ${filename} (already exists)"
        continue
    fi

    prompt_col=${PROMPT_COLUMNS[$filename]:-"query"}
    echo "Extracting: ${filename} (prompt_column=${prompt_col})"

    python src/extract_embeddings.py \
        --model_name ${MODEL_NAME} \
        --input_file ${file} \
        --prompt_column ${prompt_col} \
        --output_file ${output_file} \
        --batch_size 16 \
        --device ${DEVICE}
done

# Stage 2: Calculate local steering matrices
echo ""
echo "Stage 2: Calculating local steering matrices (K=${NUM_ANCHORS})..."
echo "========================================"

python src/calc_local_steering.py \
    --model_name llama3.1 \
    --embedding_dir ${EMBEDDING_DIR} \
    --save_path data/steering_matrix/local_steering_llama3.1_K${NUM_ANCHORS}.pt \
    --num_anchors ${NUM_ANCHORS} \
    --lambda_reg 10.0 \
    --device ${DEVICE}

echo ""
echo "Local steering matrices saved!"
echo "Path: data/steering_matrix/local_steering_llama3.1_K${NUM_ANCHORS}.pt"

# Stage 3: Generate responses
echo ""
echo "Stage 3: Generating responses..."
echo "========================================"

for config_file in config/llama3.1/*.yaml; do
    config_name=$(basename "$config_file" .yaml)
    echo "Generating: ${config_name}"

    python src/generate_response.py \
        --config_path ${config_file} \
        --model_type local_alpha \
        --local_steering_path data/steering_matrix/local_steering_llama3.1_K${NUM_ANCHORS}.pt
done

echo ""
echo "========================================"
echo "LocalAlphaSteer Pipeline Complete!"
echo "========================================"
