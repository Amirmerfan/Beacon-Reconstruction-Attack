import argparse
from scipy.spatial.distance import hamming
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
from Module import ReconstructionTool

# input values 
parser = argparse.ArgumentParser()
parser.add_argument('--beacon_size', type=int, required=True, help="Size of the beacon matrix")
parser.add_argument('--snp_count', type=int, required=True, help="Number of SNPs")
parser.add_argument('--corr_epoch', type=int, required=True, help="Number of epochs for correlation optimization")
parser.add_argument('--freq_epoch', type=int, required=True, help="Number of epochs for frequency optimization")
parser.add_argument('--path', type=str, required=True, help="Path to beacon file")
parser.add_argument('--corr_path', type=str, required=True, help="Path to correlation matrix file")


args = parser.parse_args()

# Parameters 
beacon_size = [args.beacon_size]
snp_count = args.snp_count
corr_epoch = args.corr_epoch
freq_epoch = args.freq_epoch
path = args.path
corr_path = args.corr_path

# timer
# start_time = time.time()
np.random.seed(42)
torch.manual_seed(42)

# Arrays 
precisions = []
recalls = []
f1_scores = []
accuracies = []

# data files
beacon = np.load(path)
print("Loaded beacon data...")
print(f"Beacon data loaded. Shape: {beacon.shape}")

for ind_count in beacon_size:
    print("Individual count:", ind_count)
    print("-" * 80)

    # Load 
    print("Loading correlation matrix...")
    corr1 = np.load(corr_path)
    corr1 = corr1[:snp_count, :100]

    # Calculate 
    corr_tool = ReconstructionTool(corr1)
    corr = corr_tool.calculate_correlations(corr1)
    print(corr.shape, "Initial Corr")

    # tensor 
    corr = torch.tensor(corr, dtype=torch.float32, requires_grad=True)

    # Slice 
    beacon = beacon[:snp_count, :ind_count]
    print(f"Beacon partitioning for {ind_count} individuals. Shape: {beacon.shape}")
    beacon_tool = ReconstructionTool(beacon)
    print(beacon.shape)
    number_of_ones = np.sum(beacon)
    print("Number of ones in beacon:", number_of_ones)
    print(beacon, "Initial Beacon")

    # Query
    reconstructed_beacon = beacon_tool.query(proportion=1.0)
    print(reconstructed_beacon, "Initial Reconstructed Beacon")
    print("Number of ones in reconstructed beacon:", number_of_ones)
    reconstructed_beacon = beacon_tool.compare_and_sort_columns(beacon, reconstructed_beacon)

    # Training 
    max_iterations = 10
    iteration = 0
    converged = False
    target_frequencies = torch.tensor(np.sum(beacon, axis=1), dtype=torch.float32)
    prev_reconstructed_beacon = None 

    while iteration < max_iterations:
        iteration += 1

        # Stage 1: 
        reconstructed_beacon_tensor = torch.tensor(reconstructed_beacon, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([reconstructed_beacon_tensor], lr=0.001)

        for epoch in range(corr_epoch):
            optimizer.zero_grad()
            loss = torch.norm(corr_tool.calculate_correlations(reconstructed_beacon_tensor) - corr, p='fro')
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Iteration {iteration}, Epoch {epoch}, loss {loss.item()}')

        print(reconstructed_beacon, "Reconstructed Beacon")

        # Stage 2: 
        reconstructed_beacon_tensor = torch.tensor(reconstructed_beacon, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([reconstructed_beacon_tensor], lr=0.001)

        for epoch in range(freq_epoch):
            optimizer.zero_grad()
            freq_loss = beacon_tool.frequency_loss(reconstructed_beacon_tensor, target_frequencies)
            freq_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Iteration {iteration}, Epoch {epoch}, Frequency loss: {freq_loss.item()}')

        reconstructed_beacon = (reconstructed_beacon_tensor.detach().numpy() > 0.5).astype(int)
        print(reconstructed_beacon, "Frequency-Optimized Reconstructed Beacon")

        # Convergence 
        if prev_reconstructed_beacon is not None:
            # Count the number of flips between current and previous beacons
            flips = np.sum(reconstructed_beacon != prev_reconstructed_beacon)
            print(f"Flips: {flips}")

            # Check if flips are less than 10
            if flips < 10:
                converged = True
                print("Converged based on flip.")
                break

    # Update 
    prev_reconstructed_beacon = reconstructed_beacon.copy()


    new_beacon = beacon_tool.compare_and_sort_columns(beacon, reconstructed_beacon)
    print(new_beacon, "Sorted Reconstructed Beacon")

    # Compute metrics
    accuracy = beacon_tool.calculate_accuracy(beacon, new_beacon)
    precision = beacon_tool.calculate_precision(beacon, new_beacon)
    recall = beacon_tool.calculate_recall(beacon, new_beacon)
    f1 = beacon_tool.calculate_f1(beacon, new_beacon)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Print
print("Individuals\tAccuracy\tPrecision\tRecall\t\tF1 Score")
for i in range(len(beacon_size)):
    print(f"{beacon_size[i]}\t\t{accuracies[i]:.3f}\t\t{precisions[i]:.3f}\t\t{recalls[i]:.3f}\t\t{f1_scores[i]:.3f}")