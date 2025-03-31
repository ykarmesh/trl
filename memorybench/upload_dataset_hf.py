import os
from datasets import load_from_disk
from huggingface_hub import HfApi, login

# Path to your generated dataset
dataset_dir = "/srv/flash1/yali30/code/trl/memorybench/keyframe_dataset"
dataset_name = "yali30/findingdory-val-subsampled-48"  # Choose your preferred name

def upload_dataset_to_hub():
    # Step 1: Login to HuggingFace (you'll need to have a token)
    # You can get a token from https://huggingface.co/settings/tokens
    login()  # This will prompt for your token if not already logged in
    
    # Step 2: Load the dataset
    print("Loading dataset from disk...")
    dataset = load_from_disk(os.path.join(dataset_dir, "dataset"))
    
    # Step 3: Push the dataset to the Hub
    print(f"Pushing dataset to HuggingFace Hub as {dataset_name}...")
    dataset.push_to_hub(dataset_name)
    
    # Step 4: Upload the images.zip file
    print("Uploading images.zip file...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=os.path.join(dataset_dir, "images.zip"),
        path_in_repo="images.zip",
        repo_id=dataset_name,
        repo_type="dataset"
    )
    
    print(f"Dataset successfully uploaded to HuggingFace Hub: {dataset_name}")
    print(f"You can now use it with the training script using --dataset_name {dataset_name}")

if __name__ == "__main__":
    upload_dataset_to_hub()