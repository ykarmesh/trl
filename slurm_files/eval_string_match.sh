#!/bin/bash

ROOT_DIR="/srv/flash1/yali30/code/trl/runs/apr_11"
OUTPUT_BASE_DIR="/srv/flash1/yali30/code/trl/runs/eval_string_match_apr_11"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_BASE_DIR

# Loop through each model directory
for MODEL_DIR in $ROOT_DIR/*/; do
    # Get model name from directory path
    MODEL_NAME=$(basename $MODEL_DIR)
    
    # Skip if it's the eval_string_match directory itself
    if [[ $MODEL_NAME == "eval_string_match" ]]; then
        continue
    fi
    
    # Check for checkpoint directories
    CHECKPOINT_DIRS=($MODEL_DIR/checkpoint-*)
    
    # Skip if no checkpoints found
    if [ ! -d "${CHECKPOINT_DIRS[0]}" ]; then
        echo "No checkpoints found in $MODEL_DIR, skipping..."
        continue
    fi
    
    # Process each checkpoint
    for CKPT_DIR in "${CHECKPOINT_DIRS[@]}"; do
        # Extract checkpoint ID
        CKPT_ID=$(basename $CKPT_DIR)
        
        # Create output filename
        OUTPUT_FILE="$OUTPUT_BASE_DIR/${MODEL_NAME}_${CKPT_ID}.json"
        
        echo "Processing $MODEL_NAME $CKPT_ID..."
        
        # Submit evaluation job
        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval_${MODEL_NAME}_${CKPT_ID}
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match_apr_11/${MODEL_NAME}_${CKPT_ID}-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match_apr_11/${MODEL_NAME}_${CKPT_ID}-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle
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
    --dataset_name yali30/findingdory-val-subsampled-48-qwen \
    --bf16 \
    --torch_dtype bfloat16 \
    --max_samples 500 \
    --output_file $OUTPUT_FILE
EOT
        
        # Wait a bit between submissions to avoid overwhelming the scheduler
        sleep 2
    done
done

echo "All evaluation jobs submitted!"