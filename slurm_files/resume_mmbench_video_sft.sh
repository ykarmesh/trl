#!/bin/bash
#SBATCH --job-name=mmbench_sample
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_train_apr_12/resume-48-rank1024-alpha4096-lr1e5-epoch1-3B-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_train_apr_12/resume-48-rank1024-alpha4096-lr1e5-epoch1-3B-%j.err
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
export WANDB_API_KEY=a9a6bebaaf7308fe804d6b7e35bb08bf7970cb19

source /coc/testnvme/yali30/miniforge3/etc/profile.d/conda.sh
conda deactivate
conda activate hab_memory_dev

cd /coc/testnvme/yali30/code/trl

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/sft_video_llm.py \
    --dataset_name yali30/findingdory-final-subsampled-48 \
    --dataset_train_split train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --log_level debug \
    --log_level_replica debug \
    --save_strategy steps \
    --save_steps 100 \
    --report_to wandb \
    --push_to_hub False \
    --resume_from_checkpoint=/srv/flash1/yali30/code/trl/runs/apr_12/qwen-videos-sft-48-train-rank1024-alpha4096-lr1e5-epoch1-3B/checkpoint-100 \
    --output_dir runs/apr_12/resume-qwen-videos-sft-48-train-rank1024-alpha4096-lr1e5-epoch1-3B \
    --optim adamw_torch_fused \
    --learning_rate 1e-5 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing \
    --use_liger_kernel \
    --use_peft \
    --load_in_4bit \
    --lora_r 1024 \
    --lora_alpha 4096 \
    --lora_dropout 0.1 \
    --lora_target_modules all-linear \
    --use_bnb_nested_quant \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 1 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --do_eval True \
    --eval_strategy steps \
    --eval_steps 10 \
    --eval_on_start True \
    --per_device_eval_batch_size 1 \
    --bf16_full_eval True