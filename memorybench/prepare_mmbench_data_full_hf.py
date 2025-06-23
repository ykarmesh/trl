import os
import gzip, json
import cv2
import shutil
from tqdm import tqdm
import zipfile
from datasets import Dataset, DatasetDict, Features, Value, Video
from huggingface_hub import login, HfApi, HfFolder
import pandas as pd
import getpass
import time
from datasets import disable_caching
import re
import tempfile
import shutil
import logging

# Define paths
train_video_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_train_evals_dataset_v3/interaction_videos"
train_json_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_train_evals_dataset_v3/vlm_inference_results"
val_video_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_val_evals_dataset_v3_final/interaction_videos"
val_json_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_val_evals_dataset_v3_final/vlm_inference_results"
train_valid_eps_filepath = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_train_evals_dataset_v3/episodes_to_keep.txt"
output_dir = "/coc/testnvme/yali30/code/trl/memorybench/final_arxiv_data_fixed"

# Original dataset paths
# train_dataset_path = "/srv/flash1/yali30/code/memorybench_karmesh/new_data/balanced_mmbench_dataset_v3/train/combined_episodes-with_init_and_final_poses_pddl_verified.json.gz"
# val_dataset_path = "/srv/flash1/yali30/code/memorybench_karmesh/new_data/balanced_mmbench_dataset_v3/val/final_v3-with_init_and_final_poses.json.gz"  # Update this path
# yali30/findingdory-normalized-subsampled-48

# list of valid tasks
valid_tasks_train = [
    "task_1",
    "task_2",
    "task_6",
    "task_7",
    "task_8",
    "task_9",
    "task_10",
    "task_11",
    "task_12",
    "task_13",
    "task_14",
    "task_15",
    "task_16",
    "task_17",  # 17 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_18",  # 18 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_19",
    "task_20",
    "task_21",  # 21 has XX:XX timestamp issue as it is not available in the offline dataset
    "task_22",  # 22 has XX:XX timestamp issue as it is not available in the offline dataset
    "task_23",
    "task_24",
    "task_25",
    "task_26",
    "task_27",
    "task_28",
    "task_29",
    "task_31",
    "task_32",
    "task_33",
    "task_34",
    "task_35",
    "task_36",
    "task_37",
    "task_38",
    "task_39",
    "task_40",
    # "task_41",  # 41 task was directly assigned 0 SR so we dont store oracle solution for it
    "task_42",
    "task_43",
    "task_44",
    "task_45",
    "task_46",
    "task_47",
    # "task_48", # skip for train episodes
    # "task_49", # skip for train episodes
    # "task_50", # skip for train episodes
    # "task_51", # skip for train episodes
    # "task_52", # skip for train episodes
    "task_53",
    "task_54",
    "task_55",
    "task_56",
    "task_57",
    "task_58",
    "task_59",
    "task_60",
    "task_61",
    "task_62",
    "task_63",
    "task_64",
    "task_65",
    "task_66",
    "task_67",
]

valid_tasks_val = [
    "task_1",
    "task_2",
    "task_6",
    "task_7",
    "task_8",
    "task_9",
    "task_10",
    "task_11",
    "task_12",
    "task_13",
    "task_14",
    "task_15",
    "task_16",
    "task_17",  # 17 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_18",  # 18 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_19",
    "task_20",
    "task_21",  # 21 has XX:XX timestamp issue as it is not available in the offline dataset
    "task_22",  # 22 has XX:XX timestamp issue as it is not available in the offline dataset
    "task_23",
    "task_24",
    "task_25",
    "task_26",
    "task_27",
    "task_28",
    "task_29",
    "task_31",
    "task_32",
    "task_33",
    "task_34",
    "task_35",
    "task_36",
    "task_37",
    "task_38",
    "task_39",
    "task_40",
    # "task_41",  # 41 task was directly assigned 0 SR so we dont store oracle solution for it
    "task_42",
    "task_43",
    "task_44",
    "task_45",
    "task_46",
    "task_47",
    "task_48", # skip for train episodes
    "task_49", # skip for train episodes
    "task_50", # skip for train episodes
    "task_51", # skip for train episodes
    "task_52", # skip for train episodes
    "task_53",
    "task_54",
    "task_55",
    "task_56",
    "task_57",
    "task_58",
    "task_59",
    "task_60",
    "task_61",
    "task_62",
    "task_63",
    "task_64",
    "task_65",
    "task_66",
    "task_67",
]

# Create output directories
print(f"Creating output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Create a function to process episodes and create dataset entries
def process_episodes(video_root_dir, json_root_dir, task_goals, valid_tasks, valid_eps_filepath=None):
    dataset_entries = []
    
    # Find all video files and extract episode IDs
    episode_ids = []
    episode_to_video_path = {}
    for folder_name in os.listdir(video_root_dir):
        folder_path = os.path.join(video_root_dir, folder_name)
        if os.path.isdir(folder_path):
            video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
            print(f"Found {len(video_files)} MP4 files in subfolder '{folder_name}'")
            for video_file in video_files:
                # Extract episode ID from filename (format: ep_id_X.mp4)
                match = re.match(r'ep_id_(\d+)\.mp4', video_file)
                if match:
                    episode_id = match.group(1)
                    episode_ids.append(episode_id)
                    episode_to_video_path[episode_id] = os.path.join(folder_path, video_file)
                else:
                    raise ValueError(f"Could not extract episode ID from filename: {video_file}")
    
    # skip episodes that are not in the valid episodes list
    if valid_eps_filepath is not None:
        with open(valid_eps_filepath, 'r') as file:
            valid_ep_ids = [str(line.strip()) for line in file if line.strip().isdigit()]
            episode_ids = [ep_id for ep_id in episode_ids if ep_id in valid_ep_ids]

    for ep_id in tqdm(episode_ids):
        # Skip non-directory entries
        # assert os.path.isdir(os.path.join(video_root_dir, ep_id)), f"Episode {ep_id} is not a directory"
                                        
        video_path = episode_to_video_path[ep_id]
        # Get absolute path to ensure it works from any directory
        video_path = os.path.abspath(video_path)
        
        # Verify the video file is valid (simple check - could be enhanced)
        if not os.path.exists(video_path) or not os.path.getsize(video_path) > 0:
            print(f"Invalid video file: {video_path}")
            continue
        
        # Check if corresponding JSON directory exists
        json_ep_dir = os.path.join(json_root_dir, f"ep_id_{ep_id}")
        assert os.path.exists(json_ep_dir), f"No JSON directory found for episode {ep_id}"
        
        # Define multi-goal tasks that require selecting frames for each entity
        multi_goal_tasks = ["task_12", "task_13", "task_14", "task_15", "task_16", "task_17", "task_18", "task_19", "task_20"]

        # Process each task JSON file
        for task_file in os.listdir(json_ep_dir):
            if not task_file.endswith('.json'):
                continue
            
            task_id = task_file.split('.')[0]
            # skip tasks that are not in the valid_tasks list
            if task_id not in valid_tasks:
                # print(f"skipping {task_id} because it is not in the valid_tasks list")
                continue
            
            task_path = os.path.join(json_ep_dir, task_file)
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Extract all unique keyframes from the assigns
            all_keyframes = set()
            assign_keyframes = []  # Store keyframes for each assign as a list
            assert "assigns" in task_data, f"No assigns found for task {task_id}"
            
            # Check if there's a traversal order specified
            if task_id in multi_goal_tasks:
                # Use the specified traversal order to store the individual assign keyframes
                if "assign_traversal_order" in task_data:
                    for assign_key in task_data["assign_traversal_order"]:
                        if assign_key in task_data.get("assigns", {}):
                            keyframes = task_data["assigns"][assign_key].get("keyframes", [])
                            assign_keyframes.append(sorted(keyframes))
                            all_keyframes.update(keyframes)
                else:
                    # Traverse the assigns dictionary, sort each assign keyframe list in ascending order
                    # Then append the keyframes list to assign_keyframes in ascending order of the first keyframe index
                    sorted_assigns = []
                    for assign_key, assign_data in task_data.get("assigns", {}).items():
                        keyframes = assign_data.get("keyframes", [])
                        sorted_keyframes = sorted(keyframes)
                        # Store the sorted keyframes along with their first index (or infinity if empty)
                        first_index = sorted_keyframes[0] if sorted_keyframes else float('inf')
                        sorted_assigns.append((first_index, sorted_keyframes))
                    
                    # Sort by the first keyframe index and append to assign_keyframes
                    for _, keyframes in sorted(sorted_assigns):
                        assign_keyframes.append(keyframes)
                        all_keyframes.update(keyframes)
            else:
                # No traversal order, use the order in the assigns dictionary
                for assign_key, assign_data in task_data.get("assigns", {}).items():
                    keyframes = assign_data.get("keyframes", [])
                    assign_keyframes.append(sorted(keyframes))
                    all_keyframes.update(keyframes)
            
            # Create dataset entry with appropriate output based on keyframes
            if task_id in multi_goal_tasks:
                # For multi-goal tasks, provide a list of keyframes for each entity as a list of lists
                keyframe_lists = []
                for keyframes in assign_keyframes:
                    if keyframes:
                        keyframe_lists.append(keyframes)
                    else:
                        keyframe_lists.append([])
            else:
                # For single-goal tasks, also use a list of lists format with a single inner list
                if len(all_keyframes) > 0:
                    keyframe_indices = sorted(list(all_keyframes))
                    keyframe_lists = [keyframe_indices]  # Single list inside a list
                else:
                    keyframe_lists = [[]]  # Empty list inside a list
                        
            # Use the mapped indices for the output
            output_text = json.dumps(keyframe_lists)
            if task_goals is not None:
                task_goal = task_goals[ep_id][task_id]
            else:
                task_goal = task_data["task_instruction"]
            
            # Fix specific text in task 57 goal
            if task_id == "task_57" and "object you that you" in task_goal:
                task_goal = task_goal.replace("object you that you", "object that you")
            
            # Create entry with direct video file path
            entry = {
                "id": f"ep_{ep_id}_{task_id}",
                "video": video_path,  # Direct path - no dictionary wrapper
                "qa": json.dumps([{
                    "question": f"The robot's goal is: {task_goal}",
                    "answer": output_text
                }])
            }
                            
            dataset_entries.append(entry)

    return dataset_entries

# Process train and validation data separately
print("Processing training data...")
train_entries = process_episodes(train_video_root_dir, train_json_root_dir, None, valid_tasks_train, train_valid_eps_filepath)
print("Processing validation data...")
val_entries = process_episodes(val_video_root_dir, val_json_root_dir, None, valid_tasks_val, None)

# Create a videos folder in the current directory
print(f"Output directory: {output_dir}")
print(f"Output directory exists: {os.path.exists(output_dir)}")

# Ensure output directory exists first
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

videos_folder = os.path.join(output_dir, "videos")
print(f"Videos folder path: {videos_folder}")

os.makedirs(videos_folder, exist_ok=True)
print(f"Created videos folder: {videos_folder}")
print(f"Videos folder exists: {os.path.exists(videos_folder)}")

print("Copying videos to local folder...")

# Collect all unique video paths from both train and val entries
print("Collecting unique video paths...")
all_video_paths = set()
train_video_paths = set()
val_video_paths = set()

# Count total entries vs unique videos for debugging
total_train_entries = len(train_entries)
total_val_entries = len(val_entries)

for entry in train_entries:
    all_video_paths.add(entry["video"])
    train_video_paths.add(entry["video"])

for entry in val_entries:
    all_video_paths.add(entry["video"])
    val_video_paths.add(entry["video"])

print(f"Total train entries: {total_train_entries}, Unique train videos: {len(train_video_paths)}")
print(f"Total val entries: {total_val_entries}, Unique val videos: {len(val_video_paths)}")
print(f"Total unique videos across both splits: {len(all_video_paths)}")

# Check for overlap between train and val video paths
overlap_videos = train_video_paths.intersection(val_video_paths)
if overlap_videos:
    print(f"WARNING: Found {len(overlap_videos)} videos that appear in both train and val splits")
else:
    print("No video overlap between train and val splits - good!")

# DEBUG: Limit to 10 videos from train and 10 videos from val for testing
debug_train_video_paths = list(train_video_paths)
debug_val_video_paths = list(val_video_paths)
debug_video_paths = debug_train_video_paths + debug_val_video_paths

print(f"DEBUG: Processing {len(debug_video_paths)} videos total: {len(debug_train_video_paths)} from train, {len(debug_val_video_paths)} from val")

print(f"DEBUG: {len(debug_train_video_paths)} train videos, {len(debug_val_video_paths)} val videos")

# Create train and val subfolders
train_videos_folder = os.path.join(videos_folder, "train")
val_videos_folder = os.path.join(videos_folder, "val")
os.makedirs(train_videos_folder, exist_ok=True)
os.makedirs(val_videos_folder, exist_ok=True)

# Copy videos and create a mapping from old paths to new relative paths
video_path_mapping = {}
copied_count = 0

# Copy train videos
for video_path in tqdm(debug_train_video_paths, desc="Copying train videos"):
    # Extract filename from the original path
    filename = os.path.basename(video_path)
    
    # Create new path in the train videos folder
    new_video_path = os.path.join(train_videos_folder, filename)
    
    # If filename already exists, make it unique by adding episode info
    if os.path.exists(new_video_path):
        # Extract episode info from the original path to make filename unique
        path_parts = video_path.split(os.sep)
        for part in path_parts:
            if part.startswith("ep_id_"):
                episode_info = part
                break
        else:
            episode_info = f"copy_{copied_count}"
        
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{episode_info}{ext}"
        new_video_path = os.path.join(train_videos_folder, filename)
    
    # Copy the video file
    shutil.copy2(video_path, new_video_path)
    
    # Store the mapping from old absolute path to new relative path
    relative_path = os.path.join("videos", "train", filename)
    video_path_mapping[video_path] = relative_path
    copied_count += 1

# Copy val videos
for video_path in tqdm(debug_val_video_paths, desc="Copying val videos"):
    # Extract filename from the original path
    filename = os.path.basename(video_path)
    
    # Create new path in the val videos folder
    new_video_path = os.path.join(val_videos_folder, filename)
    
    # If filename already exists, make it unique by adding episode info
    if os.path.exists(new_video_path):
        # Extract episode info from the original path to make filename unique
        path_parts = video_path.split(os.sep)
        for part in path_parts:
            if part.startswith("ep_id_"):
                episode_info = part
                break
        else:
            episode_info = f"copy_{copied_count}"
        
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{episode_info}{ext}"
        new_video_path = os.path.join(val_videos_folder, filename)
    
    # Copy the video file
    shutil.copy2(video_path, new_video_path)
    
    # Store the mapping from old absolute path to new relative path
    relative_path = os.path.join("videos", "val", filename)
    video_path_mapping[video_path] = relative_path
    copied_count += 1

print(f"Copied {copied_count} unique videos to {videos_folder} (train: {len(debug_train_video_paths)}, val: {len(debug_val_video_paths)})")

# Update video paths in dataset entries to use relative paths
# Only update entries that have videos we actually copied
print("Updating video paths to relative paths...")
updated_train_entries = []
for entry in train_entries:
    if entry["video"] in video_path_mapping:
        entry["video"] = video_path_mapping[entry["video"]]
        updated_train_entries.append(entry)

updated_val_entries = []
for entry in val_entries:
    if entry["video"] in video_path_mapping:
        entry["video"] = video_path_mapping[entry["video"]]
        updated_val_entries.append(entry)

print(f"DEBUG: Filtered to {len(updated_train_entries)} train entries and {len(updated_val_entries)} val entries")

# Use the filtered entries for dataset creation
train_entries = updated_train_entries
val_entries = updated_val_entries

# Create the dataset
print("Creating HuggingFace dataset...")

# Convert to DataFrames
train_df = pd.DataFrame(train_entries)
val_df = pd.DataFrame(val_entries)

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create the dataset dict with actual train/val splits and empty test split
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
})

# Save the dataset
dataset_dict.save_to_disk(output_dir)

print(f"Dataset created with {len(train_dataset)} training examples and {len(val_dataset)} validation examples")

# Zip the videos folder
print("Creating zip file of videos...")
videos_zip_path = os.path.join(output_dir, "videos.zip")
with zipfile.ZipFile(videos_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(videos_folder):
        for file in tqdm(files, desc="Zipping videos"):
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)
            zipf.write(file_path, arcname)

print(f"Videos zipped to {videos_zip_path}")

# Upload to HuggingFace
print("Uploading to HuggingFace...")

# Get HuggingFace token
# hf_token = input("Please enter your HuggingFace token: ")
hf_token = ""
login(token=hf_token)

# Set repository name
repo_name = input("Enter the repository name (e.g., 'username/memorybench-dataset'): ")

# Create the repository first
print("Creating repository on HuggingFace Hub...")
api = HfApi()
try:
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=False,  # Set to True if you want a private dataset
        exist_ok=True  # Don't fail if repository already exists
    )
    print(f"Repository created/verified: https://huggingface.co/datasets/{repo_name}")
except Exception as e:
    print(f"Error creating repository: {e}")
    print("Trying to continue with existing repository...")

# Push the dataset to HuggingFace Hub
print("Pushing dataset to HuggingFace Hub...")
try:
    dataset_dict.push_to_hub(
        repo_name,
        private=False,  # Set to True if you want a private dataset
        commit_message="Upload MemoryBench dataset with videos"
    )
    print("Dataset successfully pushed to HuggingFace Hub")
except Exception as e:
    print(f"Error pushing dataset: {e}")
    print("Continuing to upload videos zip file...")

# Upload the videos zip file separately using HfApi
repo_name = "yali30/" + repo_name
print("Uploading videos zip file...")
print(f"Repository ID: {repo_name}")
print(f"Zip file path: {videos_zip_path}")
print(f"Zip file exists: {os.path.exists(videos_zip_path)}")

try:
    # First, let's verify the repository exists
    try:
        repo_info = api.repo_info(repo_id=repo_name, repo_type="dataset")
        print(f"Repository verified: {repo_info.id}")
    except Exception as e:
        print(f"Repository verification failed: {e}")
        print("Repository might not exist or you don't have access to it")
        raise
    
    api.upload_file(
        path_or_fileobj=videos_zip_path,
        path_in_repo="videos.zip",
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Add videos zip file"
    )
    print("Videos zip file successfully uploaded")
except Exception as e:
    print(f"Error uploading videos zip file: {e}")
    print(f"Make sure the repository '{repo_name}' exists and you have write access to it")

print(f"Dataset and videos uploaded to HuggingFace: https://huggingface.co/datasets/{repo_name}")
print(f"Videos are available as a zip file in the repository")