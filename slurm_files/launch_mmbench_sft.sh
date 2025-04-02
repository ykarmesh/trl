#!/bin/bash
#SBATCH --job-name=mmbench_sample
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/subsample_48_sft/val-8-imgs-full_dataset-64-samples-zero2-GC-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/subsample_48_sft/val-8-imgs-full_dataset-64-samples-zero2-GC-%j.err
#SBATCH --gpus=a40:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle
#SBATCH --qos="short"
#SBATCH --partition=kira-lab
#SBATCH --requeue
#SBATCH --signal=USR1@100

MAIN_ADDR=$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

export TRANSFORMERS_CACHE=/coc/testnvme/yali30/code/trl/models
export HF_DATASETS_CACHE=/coc/testnvme/yali30/code/trl/hf_datasets

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate hab_memory_dev

cd /coc/testnvme/yali30/code/trl

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/sft_vlm_gemma3.py \
    --dataset_name yali30/findingdory-val-subsampled-48 \
    --dataset_train_split train \
    --model_name_or_path google/gemma-3-4b-it \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --output_dir runs/gemma-3-4b-it-trl-sft-memorybench-8-imgs-64-samples-GC \
    --bf16 \
    --torch_dtype bfloat16 \
    --use_peft \
    --lora_target_modules down_proj, o_proj, k_proj, q_proj, gate_proj, up_proj, v_proj \
    --gradient_checkpointing \
    --max_samples 64 \
    --max_images 8