import argparse
import numpy as np
import torch
import time
from Module import FastReconstructionTool  

parser = argparse.ArgumentParser()
parser.add_argument('--beacon_size', type=int, default=50)
parser.add_argument('--snp_count', type=int, default=1000)
parser.add_argument('--corr_epoch', type=int, default=100) 
parser.add_argument('--freq_epoch', type=int, default=100)
parser.add_argument('--path', type=str, required=True, help="Path to Target_Beacon.npy")
parser.add_argument('--corr_path', type=str, required=True, help="Path to OpenSNP.npy")
args = parser.parse_args()

print("Loading data...")
beacon_full = np.load(args.path)
corr_full = np.load(args.corr_path)


beacon = beacon_full[:args.snp_count, :args.beacon_size]
corr_data = corr_full[:args.snp_count, :100] 

print(f"Beacon Shape: {beacon.shape}")


print("Generating Baseline Guess (Algorithm 1)...")
target_frequencies = np.sum(beacon, axis=1)
baseline_guess = FastReconstructionTool.greedy_initialization(target_frequencies, args.beacon_size)

corr_tool = FastReconstructionTool(beacon.shape, np.zeros((1,1))) 
target_corr_matrix = corr_tool.calculate_current_correlations(
    torch.tensor(corr_data, dtype=torch.float32)
).numpy()

optimizer_tool = FastReconstructionTool(
    beacon.shape, 
    target_corr_matrix, 
    initial_guess=baseline_guess 
)
target_frequencies = np.sum(beacon, axis=1)

print("-" * 40)
print("Starting Optimized Reconstruction (Cyclic + Decay)...")
start_time = time.time()

max_super_iterations = 200
cycles_per_iter = 5 
corr_steps = 10
freq_steps = 5

for i in range(max_super_iterations):
    l_corr, l_freq = optimizer_tool.optimize(
        target_frequencies, 
        cycles=5, 
        corr_steps=5,   
        freq_steps=10   
    )
    
    if i % 5 == 0:
        optimizer_tool.step_schedulers()

    if i % 10 == 0:
        print(f"Iter {i}: Corr Loss={l_corr:.4f}, Freq Loss={l_freq:.4f}")

end_time = time.time()
print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

reconstructed_beacon = optimizer_tool.get_reconstruction()
reconstructed_sorted = optimizer_tool.compare_and_sort_columns(beacon, reconstructed_beacon)

acc = optimizer_tool.calculate_accuracy(beacon, reconstructed_sorted)
f1 = optimizer_tool.calculate_f1(beacon, reconstructed_sorted)

print("-" * 40)
print(f"Final Results (N={args.beacon_size}, M={args.snp_count})")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("-" * 40)