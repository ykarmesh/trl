#!/usr/bin/env python3
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
python evaluate_video_llm.py \
    --checkpoint_dir=video-llm-output \
    --dataset_name=yali30/findingdory-val-subsampled-48-qwen \
    --model_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct \
    --per_device_eval_batch_size=1 \
    --bf16 \
    --torch_dtype=bfloat16 \
    --max_samples=10 \
    --output_file=evaluation_results.json
"""

import argparse
import json
import numpy as np
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional, List
import ast

import requests
import torch
import tqdm
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import get_kbit_device_map

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def prepare_custom_dataset(example: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Prepare custom dataset example for evaluation (specifically for findingdory dataset)."""
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
    ]

    return {
        "messages": messages,
        "ground_truth": qa_pair["answer"]
    }


def parse_list_string(list_string: str) -> List[List[int]]:
    """
    Convert string representation of list of lists to actual nested list.
    Also handles cases where the list is embedded in a conversation.
    """
    try:
        # First try direct parsing
        return ast.literal_eval(list_string)
    except:
        try:
            # If direct parsing fails, try to extract the last list-like pattern
            # Look for the last occurrence of square brackets
            last_open_bracket = list_string.rfind('[')
            last_close_bracket = list_string.rfind(']')

            if last_open_bracket != -1 and last_close_bracket != -1:
                # Extract the substring that looks like a list
                potential_list = list_string[last_open_bracket:last_close_bracket + 1]

                # Try to parse the extracted list
                parsed_list = ast.literal_eval(potential_list)

                # Check if we need to wrap it in another list
                if isinstance(parsed_list, list):
                    # If the first element is an integer, wrap the whole list
                    if parsed_list and isinstance(parsed_list[0], int):
                        return [parsed_list]
                    # If the first element is a list, return as is
                    elif parsed_list and isinstance(parsed_list[0], list):
                        return parsed_list

                # If we can't determine the structure, return empty list
                print(f"Unexpected list structure in: {potential_list}")
                return []

            print(f"Failed to find list pattern in string: {list_string}")
            return []
        except:
            print(f"Failed to parse list string: {list_string}")
            return []


def calculate_relaxed_match(pred_lists: List[List[int]], gt_lists: List[List[int]]) -> float:
    """
    Calculate relaxed matching score.
    Returns 1.0 if any element in each predicted sublist appears in corresponding ground truth sublist.
    Returns 0.0 if number of sublists don't match or no elements match.
    """
    # Check if number of sublists match
    if len(pred_lists) != len(gt_lists):
        return 0.0
    
    # Check each corresponding sublist pair
    precision_all_goals = []
    for pred_sublist, gt_sublist in zip(pred_lists, gt_lists):
        # If none of the predicted elements appear in ground truth sublist, return 0
        if len(pred_sublist) == 0:
            precision = 0.0
        else:
            precision = sum(pred_elem in gt_sublist for pred_elem in pred_sublist) / len(pred_sublist)
            precision_all_goals.append(precision)

    # multiply precision of all goals
    return np.prod(precision_all_goals)

# Simple exact match metric
def calculate_exact_match(pred_text, ground_truth):
    """Calculate exact match score (1.0 if texts match exactly, 0.0 otherwise)."""
    return 1.0 if pred_text.strip() == ground_truth.strip() else 0.0


def extract_assistant_response(text):
    """Extract only the assistant's response from the full model output."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return None  # Return original text if pattern not found


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained video LLM model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--video_cache_dir", type=str, default="/tmp/videos/", help="Video cache directory")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--bf16", action="store_true", help="Use BFloat16 precision")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--torch_dtype", type=str, default=None, help="Override torch dtype")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--attn_implementation", type=str, default=None, help="Attention implementation")
    args = parser.parse_args()
    
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.torch_dtype:
        torch_dtype = getattr(torch, args.torch_dtype)
    
    print(f"Using device: {device}, dtype: {torch_dtype}")
    
    # Load base model and processor
    print(f"Loading base model from {args.model_name_or_path}...")
    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        attn_implementation=args.attn_implementation,
        use_cache=True,
    )
    
    # Load trained checkpoint
    print(f"Loading checkpoint from {args.checkpoint_dir}...")

    # Check if this is a PEFT model
    if os.path.exists(os.path.join(args.checkpoint_dir, "adapter_config.json")):
        print("Loading as PEFT model...")
        model = AutoModelForVision2Seq.from_pretrained(args.model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.checkpoint_dir)
        model = model.merge_and_unload()  # Merge adapter weights for better inference performance
    else:
        # Load as full model
        model = AutoModelForVision2Seq.from_pretrained(args.checkpoint_dir, **model_kwargs)
    
    # Put model in evaluation mode
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    
    # Load dataset
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, name=args.dataset_config, split="train")
    
    # Calculate total examples and indices to sample
    total_examples = len(dataset)
    if args.max_samples != -1:
        # Calculate step size to spread samples across dataset
        step = max(1, total_examples // args.max_samples)
        # Generate indices spread throughout the dataset
        indices = list(range(0, total_examples, step))[:args.max_samples]
        dataset = dataset.select(indices)
        print(f"Using {len(dataset)} samples spread throughout the dataset (step size: {step})")
    else:
        print(f"Using all {len(dataset)} samples for evaluation.")
    
    # Prepare dataset
    print("Preparing dataset for evaluation...")
    if "findingdory" in args.dataset_name:
        prepared_examples = [prepare_custom_dataset(example) for example in dataset]
    else:
        prepared_examples = [prepare_dataset(example, args.video_cache_dir) for example in dataset]

    # Evaluation loop
    print("Starting evaluation...")
    results, exact_match_scores, relaxed_match_scores = [], [], []
    
    for i, example in enumerate(tqdm.tqdm(prepared_examples)):
        # Process the input
        messages = example["messages"]
        print("Sample: ", example)
        ground_truth = example["ground_truth"]

        # Get video path from messages
        video_path = next(
            content["video"]
            for message in messages
            for content in message["content"]
            if content.get("type") == "video"
        )

        # Convert to model inputs
        with torch.no_grad():
            inputs = processor(
                text=processor.apply_chat_template(messages, tokenize=False),
                videos=process_vision_info(messages)[1][0],
                return_tensors="pt",
            ).to(device)
            
            # Generate output
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )

            # Decode output
            output_text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            # Extract only the assistant's response
            output_text = extract_assistant_response(output_text)

            if output_text is None:
                exact_match = 0.0
                relaxed_match = 0.0
            else:
                # Calculate exact match score
                exact_match = calculate_exact_match(output_text, ground_truth)

                # Calculate relaxed match score
                pred_lists = parse_list_string(output_text)
                gt_lists = parse_list_string(ground_truth)
                relaxed_match = calculate_relaxed_match(pred_lists, gt_lists)

            exact_match_scores.append(exact_match)
            relaxed_match_scores.append(relaxed_match)

            # Store results
            example_result = {
                "example_id": i,
                "video": os.path.basename(video_path),
                "ground_truth": ground_truth,
                "model_output": output_text,
                "exact_match": exact_match,
                "relaxed_match": relaxed_match
            }
            results.append(example_result)

            # Print results
            print(f"\nExample {i}:")
            print(f"Video: {os.path.basename(video_path)}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Output: {output_text}")
            print(f"Exact Match: {exact_match:.2f}")
            print(f"Relaxed Match: {relaxed_match:.2f}")
            print("-" * 50)

    # Calculate and print average exact match score
    avg_exact_match = sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0
    avg_relaxed_match = sum(relaxed_match_scores) / len(relaxed_match_scores) if relaxed_match_scores else 0.0
    
    print("\n" + "=" * 50)
    print(f"Overall Exact Match Score: {avg_exact_match:.4f}")
    print(f"Overall Relaxed Match Score: {avg_relaxed_match:.4f}")
    print("=" * 50)
    
    # Save results
    final_results = {
        "individual_results": results,
        "average_exact_match": avg_exact_match,
        "average_relaxed_match": avg_relaxed_match
    }
    
    print(f"Saving results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main() 