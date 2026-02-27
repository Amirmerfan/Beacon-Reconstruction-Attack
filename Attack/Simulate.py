import argparse
import numpy as np
import torch
from collections import deque
import time
from Module import ReconstructionTool  

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--beacon_size', type=int, default=50)
parser.add_argument('--snp_count', type=int, default=1000)
parser.add_argument('--corr_epoch', type=int, default=1001) 
parser.add_argument('--freq_epoch', type=int, default=501)
parser.add_argument('--path', type=str, required=True, help="Path to Target_Beacon.npy")
parser.add_argument('--corr_path', type=str, required=True, help="Path to OpenSNP.npy")
parser.add_argument('--num_runs', type=int, default=1, help="Number of independent runs")
args = parser.parse_args()

print("Loading data...")
beacon_full = (np.load(args.path) > 0).astype(np.float32)
corr_full = (np.load(args.corr_path) > 0).astype(np.float32)

beacon = beacon_full[:args.snp_count, :args.beacon_size]
corr_data = corr_full[:args.snp_count, :]

print(f"Beacon Shape: {beacon.shape}")

corr_tool = ReconstructionTool(corr_data.shape, np.zeros((1,1))) 
target_corr_matrix = corr_tool.calculate_current_correlations(
    torch.tensor(corr_data, dtype=torch.float32)
).cpu().numpy()

target_frequencies = np.mean(beacon, axis=1)

all_accuracies = []
all_f1_scores = []
all_time = []

print("-" * 50)
print(f"Starting {args.num_runs} independent optimized reconstructions...")
print(f"Configuration: {args.corr_epoch} Corr Steps / {args.freq_epoch} Freq Steps")
print("-" * 50)

for run in range(args.num_runs):
    start_time = time.time()
    
    baseline_guess = ReconstructionTool.greedy_initialization(target_frequencies, args.beacon_size)

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

        if i % 10 == 0:
            print(f"Run {run + 1} | Iteration {i} | Corr Loss: {l_corr:.6f} | Freq Loss: {l_freq:.6f} | Flips: {flips}")

        if i > grace_period and flips < 3:
            print(f"--> Optimization converged perfectly at iteration {i}. Only {flips} bits flipped.")
            break

    end_time = time.time()

    reconstructed_beacon = optimizer_tool.get_reconstruction()
    reconstructed_sorted = optimizer_tool.compare_and_sort_columns(beacon, reconstructed_beacon)
    acc = optimizer_tool.calculate_accuracy(beacon, reconstructed_sorted)
    f1 = optimizer_tool.calculate_f1(beacon, reconstructed_sorted)
    
    all_accuracies.append(acc)
    all_f1_scores.append(f1)
    all_time.append(end_time - start_time)
    
    print(f"Run {run + 1}/{args.num_runs} | Time: {end_time - start_time:.2f}s | Acc: {acc:.4f} | F1: {f1:.4f}")

print("=" * 50)
print(f"FINAL RESULTS (Averaged over {args.num_runs} runs)")
print(f"Average Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
print(f"Average Time: {np.mean(all_time):.4f} ± {np.std(all_time):.4f} Seconds")
print("=" * 50)