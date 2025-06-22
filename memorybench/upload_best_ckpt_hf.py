#!/usr/bin/env python3
"""
Script to upload a model checkpoint to HuggingFace Hub
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_checkpoint_to_hf():
    # Configuration
    checkpoint_path = "/srv/flash1/yali30/code/trl/runs/jun_6/resume-full-ft-96-frames-lr5e6-epoch5-3B/checkpoint-8800"
    repo_name = "yali30/findingdory-qwen2.5-VL-3B-finetuned"
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Uploading checkpoint from {checkpoint_path}")
    
    try:
        # Initialize HF API
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, exist_ok=True, private=False)
            print(f"Repository {repo_name} created/verified")
        except Exception as e:
            print(f"Repository creation: {e}")
        
        # Upload all files in the checkpoint directory
        print("Uploading checkpoint files...")
        
        # Get all files in checkpoint directory
        checkpoint_files = []
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, checkpoint_path)
                checkpoint_files.append((file_path, rel_path))
        
        print(f"Found {len(checkpoint_files)} files to upload")
        
        # Upload files
        for local_path, repo_path in checkpoint_files:
            print(f"Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_name,
                repo_type="model"
            )
        
        print(f"Successfully uploaded checkpoint to https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error during upload: {e}")
        return False
    
    return True

def try_transformers_upload():
    """Alternative method using transformers - only if timm issue is resolved"""
    checkpoint_path = "/srv/flash1/yali30/code/trl/runs/jun_6/resume-full-ft-96-frames-lr5e6-epoch5-3B/checkpoint-8800"
    repo_name = "yali30/findingdory-qwen2.5-VL-3B-finetuned"
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        
        print("Trying transformers approach...")
        
        # Load with torch_dtype to avoid some loading issues
        model = AutoModel.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True,
            torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        # Push to hub
        model.push_to_hub(repo_name, private=False)
        tokenizer.push_to_hub(repo_name, private=False)
        processor.push_to_hub(repo_name, private=False)
        
        return True
        
    except Exception as e:
        print(f"Transformers approach failed: {e}")
        return False

if __name__ == "__main__":
    # Make sure you're logged in to HuggingFace
    # Run: huggingface-cli login
    
    print("Attempting direct file upload approach...")
    success = upload_checkpoint_to_hf()
    
    if not success:
        print("\nDirect upload failed. Trying transformers approach...")
        success = try_transformers_upload()
    
    if success:
        print("Upload completed successfully!")
    else:
        print("Upload failed!")
