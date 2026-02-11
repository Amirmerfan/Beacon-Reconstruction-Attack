import numpy as np

# Load raw data
beacon_raw = np.load("beacon.npy")
opensnp_raw = np.load("OpenSNP.npy")

# Generate ONE permutation
num_snps = beacon_raw.shape[0]
perm = np.random.permutation(num_snps)

# Apply SAME permutation to both
beacon_shuffled = beacon_raw[perm, :]
opensnp_shuffled = opensnp_raw[perm, :]  # Critical: Must match beacon rows!

# Save them
np.save("Shuffled_Beacon.npy", beacon_shuffled)
np.save("Shuffled_OpenSNP.npy", opensnp_shuffled)