#!/bin/bash
# local_alphasteer.sh — Full pipeline for LocalAlphaSteer
#
# Steps:
#   1. Extract embeddings (same as AlphaSteer)
#   2. Compute LocalAlphaSteer parameters (K local projectors per layer)
#   3. Generate responses using LocalAlphaSteer
#
# Set K=1 to reproduce standard AlphaSteer exactly.

TRAIN_VAL_DIR=data/instructions/train_val
EMBEDDING_DIR=data/embeddings/llama3.1
NICKNAME=llama3.1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
DEVICE=cuda:0
K=4
TEMPERATURE=1.0

# Extract embeddings (skip if already done)
for file in $TRAIN_VAL_DIR/*.json; do
    filename=$(basename "$file" .json)
    echo "Extracting embeddings for $file"

    if [[ "$filename" == *"coconot"* ]]; then
        prompt_column="prompt"
    else
        prompt_column="query"
    fi

    python src/extract_embeddings.py --model_name $MODEL_NAME \
                                     --input_file $file \
                                     --prompt_column "$prompt_column" \
                                     --output_file $EMBEDDING_DIR/embeds_$filename.pt \
                                     --batch_size 16 \
                                     --device $DEVICE
done

# Compute LocalAlphaSteer parameters
LOCAL_SAVE_PATH=data/steering_matrix/local_steering_${NICKNAME}_K${K}.pt
echo "Computing LocalAlphaSteer parameters for $NICKNAME (K=$K)"
python src/calc_local_steering_matrix.py \
    --model_name $NICKNAME \
    --embedding_dir $EMBEDDING_DIR \
    --save_path $LOCAL_SAVE_PATH \
    --K $K \
    --temperature $TEMPERATURE \
    --device $DEVICE

echo "LocalAlphaSteer parameters saved to $LOCAL_SAVE_PATH"
