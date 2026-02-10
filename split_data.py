import numpy as np
import os

# 1. Load your existing full HapMap data
# Make sure this points to the file you created with convert_hapmap.py
input_file = "beacon.npy" 

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please run convert_hapmap.py first.")
    exit()

full_data = np.load(input_file)
print(f"Full Data Shape: {full_data.shape} (SNPs x Individuals)")

# 2. Split the data
# We need at least 100 people for the 'OpenSNP' reference to be good
# and 50 people for the Beacon test.
n_total = full_data.shape[1]

if n_total < 100:
    print("Warning: Your dataset is small. The reference correlation might be noisy.")

# Reserve the LAST 100 individuals as the "Reference" (fake OpenSNP)
# If we don't have enough, just split 50/50
split_point = max(n_total // 2, n_total - 100)

beacon_part = full_data[:, :split_point]      # The Victims
reference_part = full_data[:, split_point:]   # The Reference (acts as OpenSNP)

# 3. Save the files
np.save("Target_Beacon.npy", beacon_part)
np.save("OpenSNP.npy", reference_part)

print("-" * 30)
print(f"Created 'Target_Beacon.npy' with {beacon_part.shape[1]} individuals.")
print(f"Created 'OpenSNP.npy' with {reference_part.shape[1]} individuals.")
print("You can now use these in your main script.")