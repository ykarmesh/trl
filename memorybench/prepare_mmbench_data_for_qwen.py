import os
import gzip, json
import cv2
import shutil
from tqdm import tqdm
import zipfile
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi
import pandas as pd
import getpass

# Define paths
train_video_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_train_evals_dataset_v3/subsampled_dataset_96/interaction_videos"
train_json_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_train_evals_dataset_v3/subsampled_dataset_96/vlm_inference_results"
val_video_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_val_evals_dataset_v3_final/subsampled_dataset_96/interaction_videos"  # Update this path
val_json_root_dir = "/srv/flash1/yali30/code/memorybench_karmesh/runs/arxiv/oracle_val_evals_dataset_v3_final/subsampled_dataset_96/vlm_inference_results"  # Update this path
output_dir = "/coc/testnvme/yali30/code/trl/memorybench/generated_data/keyframe_dataset_qwen_train_val_96_normalized_range_v3"

# Original dataset paths
train_dataset_path = "/srv/flash1/yali30/code/memorybench_karmesh/new_data/balanced_mmbench_dataset_v3/train/combined_episodes-with_init_and_final_poses_pddl_verified.json.gz"
val_dataset_path = "/srv/flash1/yali30/code/memorybench_karmesh/new_data/balanced_mmbench_dataset_v3/val/final_v3-with_init_and_final_poses.json.gz"  # Update this path
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
os.makedirs(output_dir, exist_ok=True)

# Create a function to process episodes and create dataset entries
def process_episodes(video_root_dir, json_root_dir, task_goals, valid_tasks):
    dataset_entries = []
    
    for ep_id in tqdm(os.listdir(video_root_dir)):
        # Skip non-directory entries
        assert os.path.isdir(os.path.join(video_root_dir, ep_id)), f"Episode {ep_id} is not a directory"
        
        # Find the video file
        video_files = [f for f in os.listdir(os.path.join(video_root_dir, ep_id)) if f.endswith('.mp4')]
        if len(video_files) != 1:
            print(f"Skipping episode {ep_id} because it has {len(video_files)} video files but expected a single video file !")
            continue
        
        video_path = os.path.join(video_root_dir, ep_id, video_files[0])
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
        multi_goal_tasks = ["task_12", "task_13", "task_14", "task_15", "task_16", "task_19", "task_20"]

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
            
            # Map the original keyframe indices to the subsampled indices
            original_to_subsampled_idx = task_data["original_to_subsampled_idx"]
            mapped_keyframe_lists = []
            for sublist in keyframe_lists:
                mapped_sublist = []
                for idx in sublist:
                    if idx == -1:
                        print(f"Warning: Found idx -1 in task {task_id}, episode {ep_id}. Keeping as -1.")
                        mapped_sublist.append(-1)
                    else:
                        mapped_sublist.append(int(original_to_subsampled_idx[str(idx)]))
                mapped_keyframe_lists.append(mapped_sublist)
                        
            # Use the mapped indices for the output
            output_text = json.dumps(mapped_keyframe_lists)
            if task_goals is not None:
                task_goal = task_goals[ep_id][task_id]
            else:
                task_goal = task_data["task_instruction"]
            
            # Fix specific text in task 57 goal
            if task_id == "task_57" and "object you that you" in task_goal:
                task_goal = task_goal.replace("object you that you", "object that you")
            
            # Create entry in the new format
            entry = {
                "id": f"ep_{ep_id}_{task_id}",
                "video_path": video_path,  # Save local path to video
                "qa": json.dumps([{
                    "question": f"The robot's goal is: {task_goal}",
                    "answer": output_text
                }])
            }
                
            dataset_entries.append(entry)

    return dataset_entries

# Load and process train dataset
with gzip.open(train_dataset_path, 'rb') as f:
    train_original_dataset = json.load(f)

# Load and process validation dataset
with gzip.open(val_dataset_path, 'rb') as f:
    val_original_dataset = json.load(f)

# Create task goals dictionaries for train and val
train_task_goals = {}
for episode in train_original_dataset['episodes']:
    cur_ep_task_goal = {}
    for task_id, instr in episode['instructions'].items():
        cur_ep_task_goal[task_id] = instr['lang']
    train_task_goals[episode['episode_id']] = cur_ep_task_goal

val_task_goals = {}
for episode in val_original_dataset['episodes']:
    cur_ep_task_goal = {}
    for task_id, instr in episode['instructions'].items():
        cur_ep_task_goal[task_id] = instr['lang']
    val_task_goals[episode['episode_id']] = cur_ep_task_goal

# Process train and validation data separately
print("Processing training data...")
train_entries = process_episodes(train_video_root_dir, train_json_root_dir, None, valid_tasks_train)
print("Processing validation data...")
val_entries = process_episodes(val_video_root_dir, val_json_root_dir, None, valid_tasks_val)

# Convert to DataFrames
train_df = pd.DataFrame(train_entries)
val_df = pd.DataFrame(val_entries)

# Create Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Create empty test split
empty_test_dataset = train_dataset.select([])

# Create the dataset dict with actual train/val splits and empty test split
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": empty_test_dataset
})

# Save the dataset
dataset_dict.save_to_disk(output_dir)

print(f"Dataset created and saved to {output_dir}")
print(f"  - Train set: {len(dataset_dict['train'])} examples")
print(f"  - Validation set: {len(dataset_dict['validation'])} examples")
print(f"  - Test set: {len(dataset_dict['test'])} examples (empty)")

# Create a metadata file with information about the dataset structure
metadata = {
    "description": "Robot Task Completion Dataset",
    "format": {
        "id": "Unique identifier for the entry",
        "video_path": "Local path to the video file",
        "qa": "JSON string containing question-answer pairs"
    },
    "usage": "This dataset is compatible with the modified sft_video_llm.py script for fine-tuning video-language models."
}

with open(os.path.join(output_dir, "README.md"), 'w') as f:
    f.write(f"""# FindingDory SFT Dataset

## Description
{metadata['description']}

## Format
- id: {metadata['format']['id']}
- video_path: {metadata['format']['video_path']}
- qa: {metadata['format']['qa']}

## Usage
{metadata['usage']}

## Statistics
- Total examples: {len(train_entries) + len(val_entries)}
- Train set: {len(train_dataset)} examples
- Validation set: {len(val_dataset)} examples
- Test set: {len(empty_test_dataset)} examples (empty)
""")

print(f"Dataset README saved to {os.path.join(output_dir, 'README.md')}")

# Ask if the user wants to upload to Hugging Face Hub
upload_to_hub = input("Do you want to upload the dataset to Hugging Face Hub? (yes/no): ").lower().strip() == "yes"

if upload_to_hub:
    # Ask for the dataset name
    hf_dataset_name = input("Enter the name for your dataset on Hugging Face Hub (format: username/dataset-name): ").strip()
    
    # Ask for login token if not already logged in
    try:
        # Try to get the token, which will prompt if not logged in
        print("Please enter your Hugging Face token when prompted (or press Enter if already logged in):")
        token = getpass.getpass("Hugging Face token (will not be displayed): ")
        if token.strip():
            login(token=token)
        else:
            # If no token provided, try with stored credentials
            login(interactive=True)
            
        # Create the repository and upload
        api = HfApi()
        
        # Check if the repository already exists
        try:
            print(f"Creating dataset repository: {hf_dataset_name}")
            api.create_repo(repo_id=hf_dataset_name, repo_type="dataset")
            print(f"Repository created successfully.")
        except Exception as repo_error:
            if "You already created this dataset repo" in str(repo_error) or "409 Client Error" in str(repo_error):
                print(f"Repository {hf_dataset_name} already exists. Will upload to the existing repository.")
            else:
                raise repo_error
        
        print(f"Uploading dataset to {hf_dataset_name}...")
        dataset_dict.push_to_hub(hf_dataset_name)
        
        print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{hf_dataset_name}")
        print(f"You can use it with: --dataset_name={hf_dataset_name}")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        print("You can still use the local dataset with: --dataset_name=" + output_dir)
else:
    print("Dataset not uploaded to Hugging Face Hub.")
    print("You can use the local dataset with: --dataset_name=" + output_dir)