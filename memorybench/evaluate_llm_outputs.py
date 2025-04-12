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
    parsing_errors = 0
    
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
            
            # If either parsing returned empty list (parsing failure), count as failure
            if not gt_lists or not pred_lists:
                parsing_errors += 1
                continue
                
            # Store the actual precision score
            relaxed_score = calculate_relaxed_match(pred_lists, gt_lists)
            relaxed_scores.append(relaxed_score)
                
        except Exception as e:
            parsing_errors += 1
            print(f"Error processing example: {e}")
            continue
    
    # Calculate accuracies
    exact_accuracy = exact_matches / total_examples if total_examples > 0 else 0
    avg_relaxed_score = np.mean(relaxed_scores) if relaxed_scores else 0
    
    return {
        "total_examples": total_examples,
        "exact_matches": exact_matches,
        "avg_relaxed_score": avg_relaxed_score,
        "parsing_errors": parsing_errors,
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
        print(f"Parsing errors: {metrics['parsing_errors']}")
        print(f"Exact accuracy: {metrics['exact_accuracy']:.4f}")
    
    # Calculate average metrics across all files
    num_files = len(all_results)
    if num_files > 0:
        avg_metrics = {
            "average_exact_accuracy": sum(r["exact_accuracy"] for r in all_results) / num_files,
            "average_relaxed_score": sum(r["avg_relaxed_score"] for r in all_results) / num_files,
            "average_parsing_error_rate": sum(r["parsing_errors"] / r["total_examples"] for r in all_results) / num_files,
            "total_files_evaluated": num_files,
            "total_examples_across_files": sum(r["total_examples"] for r in all_results),
            "total_parsing_errors": sum(r["parsing_errors"] for r in all_results)
        }
    else:
        avg_metrics = {
            "average_exact_accuracy": 0.0,
            "average_relaxed_score": 0.0,
            "average_parsing_error_rate": 0.0,
            "total_files_evaluated": 0,
            "total_examples_across_files": 0,
            "total_parsing_errors": 0
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
    print(f"Total parsing errors: {avg_metrics['total_parsing_errors']}")
    print(f"Average exact accuracy: {avg_metrics['average_exact_accuracy']:.4f}")
    print(f"Average relaxed score: {avg_metrics['average_relaxed_score']:.4f}")
    print(f"Average parsing error rate: {avg_metrics['average_parsing_error_rate']:.4f}")
    print(f"\nDetailed results saved to: {output_file}")

def main():
    input_dir = "/srv/flash1/yali30/code/trl/runs/karmesh_runs"
    output_file = "/srv/flash1/yali30/code/trl/memorybench/combined_eval_results/combined_evaluation_result_karmesh.json"
    evaluate_directory(input_dir, output_file)

if __name__ == "__main__":
    main()