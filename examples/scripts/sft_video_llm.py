# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    sft_video_llm.py \
    --dataset_name=mfarre/simplevideoshorts \
    --video_cache_dir="/optional/path/to/cache/" \
    --model_name_or_path=Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size=1 \
    --output_dir=video-llm-output \
    --bf16=True \
    --tf32=True \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=4 \
    --optim="adamw_torch_fused" \
    --logging_steps=1 \
    --log_level="debug" \
    --log_level_replica="debug" \
    --save_strategy="steps" \
    --save_steps=300 \
    --learning_rate=8e-5 \
    --max_grad_norm=0.3 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type="cosine" \
    --report_to="wandb" \
    --push_to_hub=False \
    --torch_dtype=bfloat16 \
    --gradient_checkpointing=True
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

import requests
import torch
import wandb
from collections import defaultdict
from datasets import load_dataset
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def download_video(url: str, cache_dir: str) -> str:
    """Download video if not already present locally."""
    os.makedirs(cache_dir, exist_ok=True)  # Create cache dir if it doesn't exist
    filename = url.split("/")[-1]
    local_path = os.path.join(cache_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}") from e


def prepare_custom_dataset(example: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Prepare custom dataset example for training (specifically for findingdory dataset)."""
    video_path = example["video_path"]
    qa_pairs = json.loads(example["qa"])

    system_message = "You are an expert and intelligent question answering agent. You will be shown a video that was collected by a robot yesterday while navigating around a house and picking and placing objects. Each frame in the video has a unique frame index in the top left corner of the video along with the time of day information. Your job is to help the robot complete a task today by looking at the video and finding the frame indices that the robot should move to. Note: The robot uses a magic grasp action to pick up an object, where a gripper goes close to the object and the object gets magically picked up. When deciding which frame indices to choose, make sure you choose the frame indices that are closest to the object/place."

    qa_pair = qa_pairs[0]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": qa_pair["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": qa_pair["answer"]}]},
    ]

    return {"messages": messages}


def prepare_dataset(example: dict[str, Any], cache_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Prepare dataset example for training."""
    video_url = example["video_url"]
    timecoded_cc = example["timecoded_cc"]
    qa_pairs = json.loads(example["qa"])

    system_message = "You are an expert in movie narrative analysis."
    base_prompt = f"""Analyze the video and consider the following timecoded subtitles:

{timecoded_cc}

Based on this information, please answer the following questions:"""

    selected_qa = random.sample(qa_pairs, 1)[0]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": download_video(video_url, cache_dir), "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": f"{base_prompt}\n\nQuestion: {selected_qa['question']}"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": selected_qa["answer"]}]},
    ]

    return {"messages": messages}


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    video_inputs = []
    input_texts = []

    for i, example in enumerate(examples):
        video_path = next(
            content["video"]
            for message in example["messages"]
            for content in message["content"]
            if content.get("type") == "video"
        )

        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        input_texts.append("<|im_start|>".join(texts[-1].split("<|im_start|>")[:-1]))
        video_input = process_vision_info(example["messages"])[1][0]
        video_inputs.append(video_input)


    inputs = processor(text=texts, videos=video_inputs, return_tensors="pt", padding=True)
    inputs_only_tokens = processor(text=input_texts, videos=video_inputs, return_tensors="pt", padding=True)

    labels = inputs["input_ids"].clone()
    
    # Mask out padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Create a mask for input tokens using batched operations
    input_mask = inputs_only_tokens["input_ids"] != processor.tokenizer.pad_token_id
    input_lengths = input_mask.sum(dim=1)
    batch_size, seq_length = labels.shape
    mask_indices = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).to(labels.device)
    labels = torch.where(mask_indices < input_lengths.unsqueeze(1), -100, labels)
    
    # Handle visual tokens based on processor type
    visual_tokens = (
        [151652, 151653, 151656]
        if isinstance(processor, Qwen2VLProcessor)
        else [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token),
            processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        ]
    )
    
    # Mask out visual tokens
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels

    return inputs

def preprocess_logits_for_metrics(logits, labels):
    """
    Store only the argmax of the logits to save memory.
    """
    if type(logits) == tuple:
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def get_compute_metrics(processor):
    def extract_assistant_response(text):
        """Extract only the assistant's response from the full model output."""
        if "assistant\n" in text:
            return text.split("assistant\n", 1)[1].strip()
        return text  # Return original text if pattern not found

    def compute_metrics(pred):
        """
        Compute metrics for the model.
        """
        predictions = torch.tensor(pred.predictions)
        labels = torch.tensor(pred.label_ids)
        correct, total = 0, 0
        for i in range(len(predictions)):
            label_i, pred_i = labels[i], predictions[i]
            pred_i = pred_i[label_i != -100]
            label_i = label_i[label_i != -100]
        
            # Convert to text
            pred_text = processor.tokenizer.decode(pred_i, skip_special_tokens=True).strip()
            label_text = processor.tokenizer.decode(label_i, skip_special_tokens=True).strip()

            pred_text = extract_assistant_response(pred_text)
            label_text = extract_assistant_response(label_text)

            if pred_text == label_text:
                correct += 1
            total += 1

        print(pred_text.encode("utf-8"))
        print(label_text.encode("utf-8"))
        return {"accuracy": correct / total}
    return compute_metrics


@dataclass
class CustomScriptArguments(ScriptArguments):
    r"""
    Arguments for the script.

    Args:
        video_cache_dir (`str`, *optional*, defaults to `"/tmp/videos/"`):
            Video cache directory.
    """

    video_cache_dir: str = field(default="/tmp/videos/", metadata={"help": "Video cache directory."})
    train_samples: int = field(default=-1, metadata={"help": "Maximum number of samples to use for training."})
    eval_samples: int = field(default=-1, metadata={"help": "Maximum number of samples to use for evaluation."})

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")
    eval_dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="validation")

    # Setup model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(),
        # quantization_config=get_quantization_config(model_args),
        attn_implementation=model_args.attn_implementation,
        use_cache=False,
    )

    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Configure model modules for gradients
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_reentrant = False
        model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    
    # Prepare train dataset
    total_examples = len(dataset)
    if script_args.train_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // script_args.train_samples)
        # Generate indices spread throughout the datase t
        indices = list(range(0, total_examples, step))[:script_args.train_samples]
        dataset = dataset.select(indices)
        print(f"Using {len(dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(dataset)} samples for training.")

    # Filter eval dataset to specified eval_samples
    total_examples = len(eval_dataset)
    if script_args.eval_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // script_args.eval_samples)
        # Generate indices spread throughout the dataset
        indices = list(range(0, total_examples, step))[:script_args.eval_samples]
        eval_dataset = eval_dataset.select(indices)
        print(f"Using {len(eval_dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(eval_dataset)} samples for evaluation.")

    if "findingdory" in script_args.dataset_name:
        # allowed_tasks = ["1"]
        # dataset = dataset.filter(lambda x: x["id"].split('_')[-1] in allowed_tasks and x["id"].split('_')[1][0] == "5" and len(x["id"].split('_')[1]) == 3)
        # eval_dataset = eval_dataset.filter(lambda x: x["id"].split('_')[-1] in allowed_tasks and x["id"].split('_')[1][0] == "5" and len(x["id"].split('_')[1]) == 3)
        prepared_dataset = [prepare_custom_dataset(example) for example in dataset]
        prepared_eval_dataset = [prepare_custom_dataset(example) for example in eval_dataset]
    else:   
        prepared_dataset = [prepare_dataset(example, script_args.video_cache_dir) for example in dataset]
        prepared_eval_dataset = [prepare_dataset(example, script_args.video_cache_dir) for example in eval_dataset]
    
    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="memorybench_sft_video_llm")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        eval_dataset=prepared_eval_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        processing_class=processor.tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=get_compute_metrics(processor)
    )

    # Train model
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    # eval_metrics = trainer.evaluate(
    #     eval_dataset=prepared_eval_dataset
    # )
    # breakpoint()
    
    # Save final model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
