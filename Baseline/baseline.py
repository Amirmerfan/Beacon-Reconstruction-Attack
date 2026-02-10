import argparse
from scipy.spatial.distance import hamming
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import math
from Greedy import GreedyTool

# input values 
parser = argparse.ArgumentParser(description="Run Baseline Approach")
parser.add_argument("--Beacon_Size", type=int, required=True, help="Number of individuals")
parser.add_argument("--snp_count", type=int, required=True, help="Number of SNPs")
parser.add_argument("--path", type=str, required=True, help="Path to beacon file")
args = parser.parse_args()

# Parameters 
Beacon_Size = [args.Beacon_Size]
snp_count = args.snp_count
path = args.path

# Start timer and set seeds
# start_time = time.time()
np.random.seed(42)
torch.manual_seed(42)

# Arrays 
precisions = []
recalls = []
f1_scores = []
accuracies = []

# Load files
beacon = np.load(path)
print("Loaded beacon data...")
print(f"Beacon data loaded. Shape: {beacon.shape}")

for ind_count in Beacon_Size:
    print("Individual count:", ind_count)
    print("-" * 80)

    # Slicing
    beacon = beacon[snp_count:, :ind_count]
    print(f"Beacon partitioning for {ind_count} individuals. Shape: {beacon.shape}")
    beacon_tool = GreedyTool(beacon)
    print(beacon.shape)
    number_of_ones = np.sum(beacon)
    print("Number of ones in beacon:", number_of_ones)
    print(beacon, "Initial Beacon")

    # reconstruction of beacon
    reconstructed_beacon = beacon_tool.query(proportion=1.0)
    print(reconstructed_beacon, "Initial Reconstructed Beacon")
    print("Number of ones in reconstructed beacon:", number_of_ones)
    reconstructed_becon = beacon_tool.compare_and_sort_columns(beacon, reconstructed_beacon)
    # print(reconstructed_becon, "Sorted Reconstructed Beacon")

    # Computation
    accuracy = beacon_tool.calculate_accuracy(beacon, reconstructed_beacon)
    precision = beacon_tool.calculate_precision(beacon, reconstructed_beacon)
    recall = beacon_tool.calculate_recall(beacon, reconstructed_beacon)
    f1 = beacon_tool.calculate_f1(beacon, reconstructed_beacon)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Results
print("Individuals\tAccuracy\tPrecision\tRecall\t\tF1 Score")
for i in range(len(Beacon_Size)):
    print(f"{Beacon_Size[i]}\t\t{accuracies[i]:.3f}\t\t{precisions[i]:.3f}\t\t{recalls[i]:.3f}\t\t{f1_scores[i]:.3f}")
