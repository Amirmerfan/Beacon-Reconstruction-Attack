import argparse
import numpy as np
import torch
import os
import glob
import time
from Module import ReconstructionTool  
import re

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def extract_index(filename):
    match = re.search(r"Target_Beacon_(\d+)\.npy", filename)
    return int(match.group(1)) if match else -1

parser = argparse.ArgumentParser()
parser.add_argument('--beacon_size', type=int, default=50)
parser.add_argument('--snp_count', type=int, default=1000)
parser.add_argument('--corr_epoch', type=int, default=1001)
parser.add_argument('--freq_epoch', type=int, default=501)
parser.add_argument('--folder', type=str, required=True,
                    help="Folder containing Target_Beacon_i.npy and Reference_i.npy")
args = parser.parse_args()


beacon_files = glob.glob(os.path.join(args.folder, "Target_Beacon_*.npy"))
beacon_files = sorted(beacon_files, key=extract_index)

if len(beacon_files) == 0:
    print("No beacon files found!")
    exit()

print(f"Found {len(beacon_files)} beacon splits.")
print("-" * 50)

all_accuracies = []
all_f1_scores = []
all_time = []

for idx, beacon_path in enumerate(beacon_files):
    run_number = idx + 1
    if run_number == 23:
        break
    corr_path = os.path.join(args.folder, f"Reference_{run_number}.npy")

    if not os.path.exists(corr_path):
        print(f"Skipping {beacon_path} (missing matching Reference file)")
        continue

    print(f"\nProcessing Split {run_number}")
    print(f"Beacon: {beacon_path}")
    print(f"Reference: {corr_path}")

    start_time = time.time()

    beacon_full = (np.load(beacon_path) > 0).astype(np.float32)
    corr_full = (np.load(corr_path) > 0).astype(np.float32)

    beacon = beacon_full[:args.snp_count, :args.beacon_size]
    corr_data = corr_full[:args.snp_count, :]

    corr_tool = ReconstructionTool(corr_data.shape, np.zeros((1,1))) 
    target_corr_matrix = corr_tool.calculate_current_correlations(
        torch.tensor(corr_data, dtype=torch.float32)
    ).cpu().numpy()

    target_frequencies = np.mean(beacon, axis=1)

    baseline_guess = ReconstructionTool.greedy_initialization(
        target_frequencies, args.beacon_size
    )

    optimizer_tool = ReconstructionTool(
        beacon.shape, 
        target_corr_matrix, 
        initial_guess=baseline_guess 
    )

    max_iterations = 500
    grace_period = 15
    previous_binary = optimizer_tool.get_reconstruction()

    for i in range(max_iterations):
        l_corr, l_freq = optimizer_tool.optimize(
            target_frequencies,
            cycles=1,
            corr_steps=args.corr_epoch,
            freq_steps=args.freq_epoch
        )

        current_binary = optimizer_tool.get_reconstruction()
        flips = np.sum(previous_binary != current_binary)
        previous_binary = current_binary

        if i > grace_period and flips < 20:
            print(f"Converged at iteration {i}")
            break

    end_time = time.time()

    reconstructed_beacon = optimizer_tool.get_reconstruction()
    reconstructed_sorted = optimizer_tool.compare_and_sort_columns(
        beacon, reconstructed_beacon
    )

    acc = optimizer_tool.calculate_accuracy(beacon, reconstructed_sorted)
    f1 = optimizer_tool.calculate_f1(beacon, reconstructed_sorted)

    all_accuracies.append(acc)
    all_f1_scores.append(f1)
    all_time.append(end_time - start_time)

    print(f"Split {run_number} | Time: {end_time - start_time:.2f}s | Acc: {acc:.4f} | F1: {f1:.4f}")


print("\n" + "=" * 60)
print(f"FINAL RESULTS (Averaged over {len(all_accuracies)} beacon splits)")
print(f"Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
print(f"Average Time: {np.mean(all_time):.4f} ± {np.std(all_time):.4f} Seconds")
print("=" * 60)