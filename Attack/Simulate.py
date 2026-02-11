import argparse
import numpy as np
import torch
import time
from Module import FastReconstructionTool  

parser = argparse.ArgumentParser()
parser.add_argument('--beacon_size', type=int, default=50)
parser.add_argument('--snp_count', type=int, default=1000)
parser.add_argument('--corr_epoch', type=int, default=1000) # Default from paper
parser.add_argument('--freq_epoch', type=int, default=500)  # Default from paper
parser.add_argument('--path', type=str, required=True, help="Path to Target_Beacon.npy")
parser.add_argument('--corr_path', type=str, required=True, help="Path to OpenSNP.npy")
args = parser.parse_args()

print("Loading data...")
beacon_full = np.load(args.path)
corr_full = np.load(args.corr_path)

print(f"DEBUG: Original Data Range: Beacon=[{beacon_full.min()}, {beacon_full.max()}], OpenSNP=[{corr_full.min()}, {corr_full.max()}]")

beacon_full = (beacon_full > 0).astype(np.float32)
corr_full = (corr_full > 0).astype(np.float32)

print("DEBUG: Data forced to Binary (0.0 / 1.0).")

beacon = beacon_full[:args.snp_count, :args.beacon_size]
corr_data = corr_full[:args.snp_count, :100]

beacon_freqs = np.mean(beacon, axis=1)

print(f"Beacon First 5 Freqs: {beacon_freqs[:5]}")

print(f"Beacon Shape: {beacon.shape}")

print("Generating Baseline Guess (Algorithm 1)...")
target_frequencies = np.sum(beacon, axis=1)
baseline_guess = FastReconstructionTool.greedy_initialization(target_frequencies, args.beacon_size)

# Initialize Correlation Tool
corr_tool = FastReconstructionTool(corr_data.shape, np.zeros((1,1)))
# Calculate Target Correlation Matrix (Sokal-Michener)
target_corr_matrix = corr_tool.calculate_current_correlations(
    torch.tensor(corr_data, dtype=torch.float32)
).cpu().numpy()

optimizer_tool = FastReconstructionTool(
    beacon.shape, 
    target_corr_matrix, 
    initial_guess=baseline_guess 
)
target_frequencies = np.sum(beacon, axis=1)

print("-" * 40)
print(f"Starting Optimized Reconstruction (N={args.beacon_size}, M={args.snp_count})...")
print(f"Schedule: {args.corr_epoch} Corr Steps -> {args.freq_epoch} Freq Steps")
start_time = time.time()

# PAPER STRATEGY: Alternating Block Coordinate Descent
# Instead of 200 tiny steps, we do fewer "Super Iterations" with massive internal steps
max_super_iterations = 50 

for i in range(max_super_iterations):
    # Pass the ARGUMENTS into the optimize function
    l_corr, l_freq = optimizer_tool.optimize(
        target_frequencies, 
        cycles=1,                   # 1 cycle per super-loop
        corr_steps=args.corr_epoch, # Use the 1000 from arg
        freq_steps=args.freq_epoch  # Use the 500 from arg
    )
    
    # Step schedulers once per super-iteration
    #optimizer_tool.step_schedulers()

    print(f"Super-Iter {i+1}/{max_super_iterations}: Corr Loss={l_corr:.4f}, Freq Loss={l_freq:.4f}")

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