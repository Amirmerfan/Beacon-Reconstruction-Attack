import argparse
import os
import glob
import re
import time
import numpy as np
import torch
from Greedy import GreedyTool

def extract_index(filename):
    match = re.search(r"Target_Beacon_(\d+)\.npy", filename)
    return int(match.group(1)) if match else -1

parser = argparse.ArgumentParser(description="Run Baseline Approach (Batch Mode)")
parser.add_argument("--beacon_size", type=int, default=50, help="Number of individuals")
parser.add_argument("--snp_count", type=int, default=1000, help="Number of SNPs")
parser.add_argument("--folder", type=str, required=True, help="Folder containing Target_Beacon_i.npy files")
args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)

beacon_files = glob.glob(os.path.join(args.folder, "Target_Beacon_*.npy"))
beacon_files = sorted(beacon_files, key=extract_index)

if len(beacon_files) == 0:
    print("No beacon files found in the specified directory!")
    exit()

print(f"Found {len(beacon_files)} beacon splits in '{args.folder}'.")
print("-" * 60)


all_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []
all_time = []


for idx, beacon_path in enumerate(beacon_files):
    run_number = idx + 1
    
    if run_number == 23:
        break
        
    start_time = time.time()

    beacon_full = np.load(beacon_path)
    current_beacon = beacon_full[:args.snp_count, :args.beacon_size]
    
    beacon_tool = GreedyTool(current_beacon)
    
    reconstructed_beacon = beacon_tool.query(proportion=1.0)
    
    reconstructed_sorted = beacon_tool.compare_and_sort_columns(current_beacon, reconstructed_beacon)
    
    accuracy = beacon_tool.calculate_accuracy(current_beacon, reconstructed_sorted)
    precision = beacon_tool.calculate_precision(current_beacon, reconstructed_sorted)
    recall = beacon_tool.calculate_recall(current_beacon, reconstructed_sorted)
    f1 = beacon_tool.calculate_f1(current_beacon, reconstructed_sorted)

    end_time = time.time()

    
    all_accuracies.append(accuracy)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1)
    all_time.append(end_time - start_time)

    
    print(f"Split {run_number:02d} | Time: {end_time - start_time:.4f}s | Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")


print("\n" + "=" * 60)
print(f"FINAL BASELINE RESULTS (Averaged over {len(all_accuracies)} beacon splits)")
print(f"Average Accuracy:  {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"Average Precision: {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
print(f"Average Recall:    {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print(f"Average F1 Score:  {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
print(f"Average Time:      {np.mean(all_time):.4f} ± {np.std(all_time):.4f} Seconds")
print("=" * 60)