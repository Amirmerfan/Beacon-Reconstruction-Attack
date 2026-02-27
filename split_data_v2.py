import numpy as np
import os

input_file = "beacon_opensnpbed.npy"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit()

full_data = np.load(input_file)
print(f"Full Data Shape: {full_data.shape} (SNPs x Individuals)")

n_total = full_data.shape[1]

if n_total < 100:
    print("Warning: Dataset is small. Some windows may not be created.")

beacon_size = 50
reference_size = 100
step = beacon_size

iteration = 1
start = 0

while True:
    beacon_start = start
    beacon_end = beacon_start + beacon_size

    reference_start = beacon_end
    reference_end = reference_start + reference_size

    if beacon_end > n_total:
        break

    if reference_start >= n_total:
        break

    beacon_part = full_data[:, beacon_start:beacon_end]
    reference_part = full_data[:, reference_start:min(reference_end, n_total)]

    beacon_filename = f"Target_Beacon_{iteration}.npy"
    reference_filename = f"Reference_{iteration}.npy"

    np.save(beacon_filename, beacon_part)
    np.save(reference_filename, reference_part)

    print("-" * 40)
    print(f"Iteration {iteration}")
    print(f"Beacon: Individuals {beacon_start+1} to {beacon_end}")
    print(f"Reference: Individuals {reference_start+1} to {min(reference_end, n_total)}")
    print(f"Saved {beacon_filename} and {reference_filename}")

    start += step
    iteration += 1

print("\nDone. Beacon/reference pairs created.")