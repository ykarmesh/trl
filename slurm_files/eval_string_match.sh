#!/bin/bash
#SBATCH --job-name=mmbench_sample
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match/eval_string_match-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match/eval_string_match-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle
#SBATCH --qos="short"
#SBATCH --partition=kira-lab,overcap
#SBATCH --requeue
#SBATCH --signal=USR1@100

MAIN_ADDR=$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

# Define variables at the top, before SBATCH directives
CKPT_NAME=qwen-videos-sft-48-train-full-ft-lr8e5-epoch5-3B
CKPT_DIR=/srv/flash1/yali30/code/trl/runs/qwen-videos-sft-48-train-full-ft-lr8e5-epoch5-3B/checkpoint-850

# Extract checkpoint ID from CKPT_DIR
CKPT_ID=$(basename $CKPT_DIR)  # This will extract 'checkpoint-50'

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
    --output_file runs/eval_string_match/${CKPT_NAME}_${CKPT_ID}.json
