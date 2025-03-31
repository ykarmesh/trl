import os
import gzip, json
import cv2
import shutil
from tqdm import tqdm
import zipfile
from datasets import Dataset
from huggingface_hub import HfApi

# Define paths
video_root_dir = "/srv/flash1/yali30/code/memorybench_dev/runs/mmbench_evals/oracle_all_task_final_eval/subsampled_dataset_48/interaction_videos"
json_root_dir = "/srv/flash1/yali30/code/memorybench_dev/runs/mmbench_evals/oracle_all_task_final_eval/subsampled_dataset_48/vlm_inference_results"
output_dir = "/srv/flash1/yali30/code/trl/memorybench/keyframe_dataset"
images_dir = os.path.join(output_dir, "images")
original_dataset_path = "/srv/flash1/kyadav32/code/gunshi/memorybench/memorybench/data/datasets/hssd/memory_dataset/balanced_mmbench_dataset_v2/val/combined_episodes_with_pddl_subset_cleaned_all_instr_5.json.gz"

# list of valid tasks
valid_tasks = [
    "task_1",
    "task_2",
    "task_6",
    "task_7",
    "task_8",
    "task_9",
    "task_10",
    "task_12",
    "task_13",
    "task_14",
    "task_15",
    "task_16",
    "task_17",  # 17 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_18",  # 18 is sequential task with receptacles and has some issue so the oracle solution doesnt exist currently
    "task_19",
    "task_20",
    # "task_21",  # 21 has XX:XX timestamp issue as it is not available in the offline dataset
    # "task_22",  # 22 has XX:XX timestamp issue as it is not available in the offline dataset
    "task_24",
    "task_25",
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
    "task_41",  # 41 task was directly assigned 0 SR so we dont store oracle solution for it
    "task_44",
    "task_45",
    "task_46",
    "task_47",
    "task_48",
    "task_49",
    "task_50",
    "task_51",
    "task_52",
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
]

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Load the original dataset
with gzip.open(original_dataset_path, 'rb') as f:
    original_dataset = json.load(f)

# Create a dictionary to store task goals
task_goals = {}

# Process the original dataset to extract task goals
for episode in original_dataset['episodes']:
    cur_ep_task_goal = {}

    for task_id,instr in episode['instructions'].items():
        cur_ep_task_goal[task_id] = instr['lang']           # task_id is string like "task_1", instr['lang'] is the task goal string   

    task_goals[episode['episode_id']] = cur_ep_task_goal
    
# Function to extract all frames from a video
def extract_all_frames(video_path, output_prefix):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_paths = []
    
    for idx in range(frame_count):
        # Set the position of the video file to the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = f"{output_prefix}_frame_{idx}.jpg"
            full_path = os.path.join(images_dir, frame_path)
            cv2.imwrite(full_path, frame)
            frame_paths.append(os.path.join("images", frame_path))
        else:
            print(f"Failed to extract frame {idx} from {video_path}")
    
    cap.release()
    return frame_paths

# Process all episodes and tasks
dataset_entries = []
processed_videos = {}  # Cache to avoid re-extracting frames from the same video

for ep_id in tqdm(os.listdir(video_root_dir)):
    # Skip non-directory entries
    assert os.path.isdir(os.path.join(video_root_dir, ep_id)), f"Episode {ep_id} is not a directory"
    
    # Find the video file
    video_files = [f for f in os.listdir(os.path.join(video_root_dir, ep_id)) if f.endswith('.mp4')]
    assert len(video_files) == 1, f"Expected 1 video file for episode {ep_id}, found {len(video_files)}"
    
    video_path = os.path.join(video_root_dir, ep_id, video_files[0])
    
    # Extract all frames from the video (only once per video)
    if video_path not in processed_videos:
        frame_paths = extract_all_frames(video_path, f"ep_{ep_id}")
        processed_videos[video_path] = frame_paths
    else:
        frame_paths = processed_videos[video_path]
    
    if not frame_paths:
        print(f"Failed to extract frames from {video_path}")
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
            print(f"skipping {task_id} because it is not in the valid_tasks list")
            continue
        
        task_path = os.path.join(json_ep_dir, task_file)
        with open(task_path, 'r') as f:
            task_data = json.load(f)
        
        # Extract all unique keyframes from the assigns
        all_keyframes = set()
        assign_keyframes = []  # Store keyframes for each assign as a list
        assert "assigns" in task_data, f"No assigns found for task {task_id}"
        
        # Check if there's a traversal order specified
        if "assign_traversal_order" in task_data:
            # Use the specified traversal order
            for assign_key in task_data["assign_traversal_order"]:
                if assign_key in task_data.get("assigns", {}):
                    keyframes = task_data["assigns"][assign_key].get("keyframes", [])
                    assign_keyframes.append(sorted(keyframes))
                    all_keyframes.update(keyframes)
        else:
            # No traversal order, use the order in the assigns dictionary
            for assign_key, assign_data in task_data.get("assigns", {}).items():
                keyframes = assign_data.get("keyframes", [])
                assign_keyframes.append(sorted(keyframes))
                all_keyframes.update(keyframes)
        
        # Create dataset entry with appropriate output based on keyframes
        if task_id in multi_goal_tasks:
            # For multi-goal tasks, provide a list of keyframes for each entity without IDs
            output_lines = ["Solution keyframes for each entity:"]
            for i, keyframes in enumerate(assign_keyframes, 1):
                if keyframes:
                    output_lines.append(f"Entity {i}: {', '.join([str(idx) for idx in keyframes])}")
                else:
                    output_lines.append(f"Entity {i}: No valid keyframe")
            output_text = "\n".join(output_lines)
        else:
            # For single-goal tasks, the model can choose any valid keyframe
            if len(all_keyframes) > 0:
                keyframe_indices = sorted(list(all_keyframes))
                output_text = f"The solution keyframes are: {', '.join([str(idx) for idx in keyframe_indices])}."
            else:
                output_text = "No frame solves the task."
        
        task_goal = task_goals[ep_id][task_id]
        
        entry = {
            "question": f"The robot's goal is: {task_goal}",
            "input_image_path": frame_paths,  # All frames from the video
            "context": f"""You are an expert and intelligent question answering agent. You will be shown a video that was collected by a robot yesterday while navigating around a house and picking and placing objects. Each frame in the video has a unique frame index in the top left corner of the video along with the time of day information. Your job is to help the robot complete a task today by looking at the video and finding the frame indices that the robot should move to. Note: The robot uses a magic grasp action to pick up an object, where a gripper goes close to the object and the object gets magically picked up. When deciding which frame indices to choose, make sure you choose the frame indices that are closest to the object/place.""",
            "output": output_text
        }
                
        dataset_entries.append(entry)
        
# Create dataset dictionary
dataset_dict = {
    "question": [],
    "input_image_path": [],
    "context": [],
    "output": []
}

for entry in dataset_entries:
    dataset_dict["question"].append(entry["question"])
    dataset_dict["input_image_path"].append(entry["input_image_path"])
    dataset_dict["context"].append(entry["context"])
    dataset_dict["output"].append(entry["output"])

# Create and save the dataset
dataset = Dataset.from_dict(dataset_dict)
dataset.save_to_disk(os.path.join(output_dir, "dataset"))

# Create a zip file of the images
with zipfile.ZipFile(os.path.join(output_dir, "images.zip"), "w") as zipf:
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            zipf.write(os.path.join(root, file), 
                      arcname=os.path.join("images", file))

print(f"Dataset created with {len(dataset_entries)} entries")