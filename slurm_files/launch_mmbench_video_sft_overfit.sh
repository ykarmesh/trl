#!/bin/bash
#SBATCH --job-name=mmbench_sample
#SBATCH --output=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_overfit_apr_18/with-liger-overfit-%j.out
#SBATCH --error=/coc/testnvme/yali30/code/trl/slurm_logs/sft_qwen_overfit_apr_18/with-liger-overfit-%j.err
#SBATCH --gpus=a40:8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=gideon,irona,calculon,bb8,walle,xaea-12
#SBATCH --qos="short"
#SBATCH --partition=kira-lab,overcap
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

PORT=$((29500 + ${SLURM_JOB_ID} % 10000))

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_video_llm.py \
    --dataset_name yali30/findingdory-final-subsampled-48 \
    --dataset_train_split train \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 100 \
    --push_to_hub False \
    --logging_steps 1 \
    --log_level debug \
    --log_level_replica debug \
    --save_strategy steps \
    --save_steps 100 \
    --report_to wandb \
    --push_to_hub False \
    --output_dir runs/with-liger-overfit-qwen-videos-sft-48-frames-3B \
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
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 1 \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory True \
    --do_eval True \
    --eval_strategy steps \
    --eval_steps 10 \
    --eval_on_start True \
    --per_device_eval_batch_size 1 \
    --bf16_full_eval True \
    --use_liger_kernel
    # --main_process_port ${PORT} \
    # --use_peft \
    # --lora_r 16 \
    # --lora_alpha 128 \
    # --lora_dropout 0.0 \
    # --lora_target_modules all-linear \
    # --max_samples 100 \

# --load_in_4bit \
# --use_bnb_nested_quant \