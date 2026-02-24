import numpy as np
import os


input_file = "beacon_new.npy" 

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run convert_hapmap.py first.")
    exit()

full_data = np.load(input_file)
print(f"Full Data Shape: {full_data.shape} (SNPs x Individuals)")

n_total = full_data.shape[1]

if n_total < 100:
    print("Warning: Your dataset is small. The reference correlation might be noisy.")


beacon_part = full_data[:, :50]
reference_part = full_data[:, 50:150]

np.save("Target_Beacon.npy", beacon_part)
np.save("OpenSNP.npy", reference_part)

print("-" * 30)
print(f"Created 'Target_Beacon.npy' with {beacon_part.shape[1]} individuals.")
print(f"Created 'OpenSNP.npy' with {reference_part.shape[1]} individuals.")
print("You can now use these in your main script.")