import json
import ast
import os
from pathlib import Path
from typing import List, Dict
import numpy as np

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
    Calculate relaxed matching score as a product of precision scores for each sublist.
    
    For each predicted sublist and corresponding ground truth sublist:
    - Calculates precision as (number of predicted elements in ground truth) / (number of predicted elements)
    - Returns product of precision scores across all sublists
    
    Returns:
    - 0.0 if number of sublists don't match
    - Product of precision scores (between 0.0 and 1.0) otherwise
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

def evaluate_results(results_file: str) -> Dict:
    """Evaluate both exact and relaxed matching metrics from results file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    total_examples = len(data["individual_results"])
    exact_matches = 0
    relaxed_scores = []  # Changed from counter to list of scores
    
    for result in data["individual_results"]:
        try:
            # Get ground truth and model output
            gt_string = result["ground_truth"]
            pred_string = result["model_output"]
            
            # Calculate exact match
            if gt_string == pred_string:
                exact_matches += 1
            
            # Calculate relaxed match
            gt_lists = parse_list_string(gt_string)
            pred_lists = parse_list_string(pred_string)
            
            # If either parsing returned empty list (parsing failure), treat as 0 for relaxed metric
            if not gt_lists or not pred_lists:
                relaxed_scores.append(0.0)
                continue
                
            # Store the actual precision score
            relaxed_score = calculate_relaxed_match(pred_lists, gt_lists)
            relaxed_scores.append(relaxed_score)
                
        except Exception as e:
            # Treat exceptions as 0 for relaxed metric
            relaxed_scores.append(0.0)
            print(f"Error processing example: {e}")
            continue
    
    # Calculate accuracies
    exact_accuracy = exact_matches / total_examples if total_examples > 0 else 0
    avg_relaxed_score = np.mean(relaxed_scores) if relaxed_scores else 0
    
    return {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "avg_relaxed_score": avg_relaxed_score,
        "exact_accuracy": exact_accuracy
    }

def evaluate_directory(input_dir: str, output_file: str):
    """
    Evaluate all JSON files in the directory and save combined results.
    """
    # Get all JSON files in the directory
    json_files = list(Path(input_dir).glob('*.json'))
    print(f"Found {len(json_files)} JSON files to evaluate")
    
    # Skip the combined results file if it exists in the directory
    json_files = [f for f in json_files if f.name != os.path.basename(output_file)]
    
    # Evaluate each file
    all_results = []
    for json_file in json_files:
        print(f"\nEvaluating {json_file.name}...")
        metrics = evaluate_results(str(json_file))
        
        # Add filename to metrics
        metrics["file_name"] = json_file.name
        all_results.append(metrics)
        
        # Print individual file results
        print(f"Results for {json_file.name}:")
        print(f"Total examples: {metrics['total_examples']}")
        print(f"Exact matches: {metrics['exact_matches']}")
        print(f"Average relaxed score: {metrics['avg_relaxed_score']:.4f}")
        print(f"Exact accuracy: {metrics['exact_accuracy']:.4f}")
    
    # Plot the results
    print("\nCreating accuracy comparison plot...")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract data for plotting and clean up file names
        results_with_names = []
        for result in all_results:
            name = result["file_name"]
            if name.startswith("no_liger_epoch4_ckpt_2400"):
                name = name.replace("no_liger_epoch4_ckpt_2400_", "")
            
            # Extract sample number from filename (assumes format like "500_sample_val.json")
            sample_num = 0
            try:
                sample_num = int(name.split("_")[0])
            except (ValueError, IndexError):
                pass
                
            results_with_names.append((sample_num, name, result))
        
        # Sort results by sample number
        results_with_names.sort()  # Sorts by first element (sample_num)
        
        # Extract the sorted data
        file_names = [item[1] for item in results_with_names]
        exact_accuracies = [item[2]["exact_accuracy"] for item in results_with_names]
        relaxed_accuracies = [item[2]["avg_relaxed_score"] for item in results_with_names]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set width of bars
        bar_width = 0.35
        x = np.arange(len(file_names))
        
        # Create bars
        ax.bar(x - bar_width/2, exact_accuracies, bar_width, label='Exact Accuracy')
        ax.bar(x + bar_width/2, relaxed_accuracies, bar_width, label='Relaxed Accuracy')
        
        # Add labels, title and legend
        ax.set_xlabel('Number of evaluation samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Comparison of Exact and Relaxed Accuracy by Number of Evaluation Samples')
        ax.set_xticks(x)
        ax.set_xticklabels(file_names, rotation=45, ha='right')
        ax.legend()
        
        # Adjust layout and save
        fig.tight_layout()
        plt.savefig('/srv/flash1/yali30/code/trl/runs/eval_string_match_apr_21/accuracy_comparison.png')
        print("Plot saved as 'accuracy_comparison.png'")
    except ImportError:
        print("Could not create plot: matplotlib is not installed")
    
    # Calculate average metrics across all files
    num_files = len(all_results)
    if num_files > 0:
        avg_metrics = {
            "average_exact_accuracy": sum(r["exact_accuracy"] for r in all_results) / num_files,
            "average_relaxed_score": sum(r["avg_relaxed_score"] for r in all_results) / num_files,
            "total_files_evaluated": num_files,
            "total_examples_across_files": sum(r["total_examples"] for r in all_results),
        }
    else:
        avg_metrics = {
            "average_exact_accuracy": 0.0,
            "average_relaxed_score": 0.0,
            "total_files_evaluated": 0,
            "total_examples_across_files": 0,
        }
    
    # Combine all results
    final_results = {
        "individual_file_results": all_results,
        "average_metrics": avg_metrics
    }
    
    # Save combined results
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Total files evaluated: {avg_metrics['total_files_evaluated']}")
    print(f"Total examples across all files: {avg_metrics['total_examples_across_files']}")
    print(f"Average exact accuracy: {avg_metrics['average_exact_accuracy']:.4f}")
    print(f"Average relaxed score: {avg_metrics['average_relaxed_score']:.4f}")
    print(f"\nDetailed results saved to: {output_file}")

def main():
    input_dir = "/srv/flash1/yali30/code/trl/runs/eval_string_match_apr_21"
    output_file = "/srv/flash1/yali30/code/trl/runs/eval_string_match_apr_21/aggregated_results.json"
    evaluate_directory(input_dir, output_file)

if __name__ == "__main__":
    main()