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
parser.add_argument('--corr_epoch', type=int, default=1000) 
parser.add_argument('--freq_epoch', type=int, default=500)
parser.add_argument('--path', type=str, required=True, help="Path to Target_Beacon.npy")
parser.add_argument('--corr_path', type=str, required=True, help="Path to OpenSNP.npy")
parser.add_argument('--num_runs', type=int, default=1, help="Number of independent runs")
args = parser.parse_args()

print("Loading data...")
beacon_full = (np.load(args.path) > 0).astype(np.float32)
corr_full = (np.load(args.corr_path) > 0).astype(np.float32)

beacon = beacon_full[:args.snp_count, :args.beacon_size]
corr_data = corr_full[:args.snp_count, :100]

print(f"Beacon Shape: {beacon.shape}")

corr_tool = ReconstructionTool(corr_data.shape, np.zeros((1,1))) 
target_corr_matrix = corr_tool.calculate_current_correlations(
    torch.tensor(corr_data, dtype=torch.float32)
).cpu().numpy()

target_frequencies = np.sum(beacon, axis=1)

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

    max_iterations = 1001
    min_iterations = 21

    window_size = 10
    corr_history = deque(maxlen=window_size)
    freq_history = deque(maxlen=window_size)

    tolerance_corr = 0.001
    tolerance_freq = 0.0001

    current_corr_steps = args.corr_epoch
    current_freq_steps = args.freq_epoch

    corr_active = True
    freq_active = True

    for i in range(max_iterations):
        l_corr, l_freq = optimizer_tool.optimize(
            target_frequencies, 
            cycles=1,
            corr_steps=current_corr_steps, 
            freq_steps=current_freq_steps
        )

        if corr_active:
            corr_history.append(l_corr)
        if freq_active:
            freq_history.append(l_freq)

        if i > min_iterations:
            if corr_active and len(corr_history) == window_size:
                corr_progress = corr_history[0] - corr_history[-1]
                if abs(corr_progress) < tolerance_corr:
                    corr_active = False
                    current_corr_steps = 0

            if freq_active and len(freq_history) == window_size:
                freq_progress = freq_history[0] - freq_history[-1]
                if abs(freq_progress) < tolerance_freq:
                    freq_active = False
                    current_freq_steps = 0

            if not corr_active and not freq_active:
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
print(f"Average Time: {np.mean(all_time):.4f} ± {np.std(all_time):.4f}")
print("=" * 50)