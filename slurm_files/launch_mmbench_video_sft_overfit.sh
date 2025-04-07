#!/bin/bash
#SBATCH --job-name=mmbench_sample
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen/patched-overfit-100-val-48-imgs-rank16-alpha128-maxGrad1-noLoraDropout-no4bit-const_lr_2e4-3B-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen/patched-overfit-100-val-48-imgs-rank16-alpha128-maxGrad1-noLoraDropout-no4bit-const_lr_2e4-3B-%j.err
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle,puma,samantha
#SBATCH --qos="short"
#SBATCH --partition=kira-lab
#SBATCH --requeue
#SBATCH --signal=USR1@100

MAIN_ADDR=$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

export TRANSFORMERS_CACHE=/coc/testnvme/yali30/code/trl/models
export HF_DATASETS_CACHE=/coc/testnvme/yali30/code/trl/hf_datasets
export WANDB_API_KEY=

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate hab_memory_dev

cd /coc/testnvme/yali30/code/trl

PORT=$((29500 + ${SLURM_JOB_ID} % 10000))

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2_single_gpu.yaml \
    --main_process_port ${PORT} \
    examples/scripts/sft_video_llm.py \
    --dataset_name yali30/findingdory-val-subsampled-48-qwen \
    --dataset_train_split train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 50 \
    --push_to_hub False \
    --logging_steps 1 \
    --log_level debug \
    --log_level_replica debug \
    --save_strategy steps \
    --save_steps 100 \
    --report_to wandb \
    --push_to_hub False \
    --output_dir runs/qwen-videos-sft-48-frames-liger-patched-overfit-100-rank16-alpha128-maxGrad1-noLoraDropout-no4bit-const_lr_2e4-3B \
    --optim adamw_torch_fused \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type constant \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 128 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --use_liger_kernel \
    --max_samples 100 \
    --gradient_checkpointing

# --load_in_4bit \
# --use_bnb_nested_quant \