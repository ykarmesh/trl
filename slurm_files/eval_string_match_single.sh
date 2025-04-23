#!/bin/bash
#SBATCH --job-name=mmbench_val
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match_apr_21/val-5000-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/eval_string_match_apr_21/val-5000-%j.err
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
    --checkpoint_dir /srv/flash1/yali30/code/trl/runs/apr_22/no-liger-full-ft-lr6e5-epoch4-correct-3B/checkpoint-6 \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name yali30/findingdory-final-subsampled-48 \
    --max_samples 5000 \
    --per_device_eval_batch_size 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --output_file /srv/flash1/yali30/code/trl/runs/eval_string_match_apr_21/no_liger_epoch4_ckpt_2400_5000_sample_val.json \
    --split validation