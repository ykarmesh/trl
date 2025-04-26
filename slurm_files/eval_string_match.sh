#!/bin/bash

ROOT_DIR="/srv/flash1/yali30/code/trl/runs/apr_22"

# Debug option - set DEBUG_MODE to true and specify a directory to only process that one
DEBUG_MODE=${DEBUG_MODE:-false}
DEBUG_DIR=${DEBUG_DIR:-""}

# Allow direct specification of directories to process
SPECIFIC_DIRS=${SPECIFIC_DIRS:-""}

# Minimum checkpoint number to consider (e.g., 1000 to only evaluate checkpoints >= checkpoint-1000)
MIN_CHECKPOINT=${MIN_CHECKPOINT:-0}

if [ -n "$SPECIFIC_DIRS" ]; then
    echo "Processing specific directories: $SPECIFIC_DIRS"
    # Split the comma-separated string into an array
    IFS=',' read -ra DIRS_TO_PROCESS <<< "$SPECIFIC_DIRS"
elif [ "$DEBUG_MODE" = true ] && [ -n "$DEBUG_DIR" ]; then
    echo "Debug mode enabled. Only processing directory: $DEBUG_DIR"
    # Set ROOT_DIR to the debug directory
    ROOT_DIR=$DEBUG_DIR
    # Process only the root directory itself
    DIRS_TO_PROCESS=("$ROOT_DIR")
else
    # Normal mode - process all directories in ROOT_DIR
    DIRS_TO_PROCESS=($ROOT_DIR/*)
fi

echo "Minimum checkpoint to consider: $MIN_CHECKPOINT"

# Loop through each model directory
for MODEL_DIR in "${DIRS_TO_PROCESS[@]}"; do
    # Get model name from directory path
    MODEL_NAME=$(basename $MODEL_DIR)
    
    # Skip if it's the eval_string_match directory itself
    if [[ $MODEL_NAME == "eval_string_match" ]]; then
        continue
    fi
    
    echo "Processing model directory: $MODEL_DIR"
    
    # Find all checkpoint directories
    ALL_CHECKPOINT_DIRS=($MODEL_DIR/checkpoint-*)
    
    # Skip if no checkpoints found
    if [ ! -d "${ALL_CHECKPOINT_DIRS[0]}" ]; then
        echo "No checkpoints found in $MODEL_DIR, skipping..."
        continue
    fi
    
    # Filter checkpoints based on minimum checkpoint number
    CHECKPOINT_DIRS=()
    for CHECKPOINT_DIR in "${ALL_CHECKPOINT_DIRS[@]}"; do
        # Extract the checkpoint number
        CHECKPOINT_NUM=$(basename "$CHECKPOINT_DIR" | sed 's/checkpoint-//')
        
        # Skip checkpoints with names that don't contain numbers (e.g., checkpoint-latest)
        if ! [[ "$CHECKPOINT_NUM" =~ ^[0-9]+$ ]]; then
            echo "Skipping non-numeric checkpoint: $CHECKPOINT_DIR"
            continue
        fi
        
        # Compare with minimum checkpoint number
        if [ "$CHECKPOINT_NUM" -gt "$MIN_CHECKPOINT" ]; then
            CHECKPOINT_DIRS+=("$CHECKPOINT_DIR")
        else
            echo "Skipping checkpoint-$CHECKPOINT_NUM (not greater than minimum threshold)"
        fi
    done
    
    # Check if we have any checkpoints after filtering
    if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
        echo "No checkpoints meeting the minimum threshold found in $MODEL_DIR, skipping..."
        continue
    fi
    
    echo "Found ${#CHECKPOINT_DIRS[@]} checkpoints meeting criteria in $MODEL_DIR"
    
    # Process each checkpoint
    for CKPT_DIR in "${CHECKPOINT_DIRS[@]}"; do
        # Extract checkpoint ID
        CKPT_ID=$(basename $CKPT_DIR)
        
        # Create output filename
        OUTPUT_FILE="$MODEL_DIR/evals/${CKPT_ID}.json"

        # Create output dir
        OUTPUT_BASE_DIR=$(dirname $OUTPUT_FILE)
        mkdir -p $OUTPUT_BASE_DIR
        
        echo "Processing $MODEL_NAME $CKPT_ID..."
        echo "CKPT_DIR: $CKPT_DIR --- OUTPUT_FILE: $OUTPUT_FILE"
        
        # Submit evaluation job
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval_mmbench
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_train_apr_22/evals/${MODEL_NAME}/${CKPT_ID}-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_train_apr_22/evals/${MODEL_NAME}/${CKPT_ID}-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle,puma,xaea-12
#SBATCH --qos="short"
#SBATCH --partition=kira-lab,overcap
#SBATCH --requeue
#SBATCH --signal=USR1@100

export TRANSFORMERS_CACHE=/coc/testnvme/yali30/code/trl/models
export HF_DATASETS_CACHE=/coc/testnvme/yali30/code/trl/hf_datasets
export WANDB_API_KEY=a9a6bebaaf7308fe804d6b7e35bb08bf7970cb19

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate hab_memory_dev

cd /coc/testnvme/yali30/code/trl

srun -u python examples/scripts/evaluate_video_llm.py \
    --checkpoint_dir $CKPT_DIR \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name yali30/findingdory-normalized-subsampled-48 \
    --bf16 \
    --torch_dtype bfloat16 \
    --output_file $OUTPUT_FILE \
    --split validation
EOT
        
        # Wait a bit between submissions to avoid overwhelming the scheduler
        sleep 2
    done
done

echo "All evaluation jobs submitted!"