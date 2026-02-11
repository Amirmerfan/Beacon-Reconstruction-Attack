import numpy as np
import os

# CONFIGURATION
# ---------------------------------------------------------
BEACON_FILE = "beacon.npy"      # Your ORIGINAL, UNSHUFFLED beacon file
OPENSNP_FILE = "OpenSNP.npy"    # Your ORIGINAL, UNSHUFFLED correlation file
OUTPUT_BEACON = "Aligned_Beacon.npy"
OUTPUT_OPENSNP = "Aligned_OpenSNP.npy"
# ---------------------------------------------------------

def main():
    print(f"Loading {BEACON_FILE} and {OPENSNP_FILE}...")
    
    if not os.path.exists(BEACON_FILE) or not os.path.exists(OPENSNP_FILE):
        print("ERROR: Could not find original files. Please check filenames.")
        return

    beacon = np.load(BEACON_FILE)
    opensnp = np.load(OPENSNP_FILE)
    
    # 1. Validation
    # We only care that the number of SNPs (rows) matches
    rows_beacon = beacon.shape[0]
    rows_opensnp = opensnp.shape[0]
    
    # Trim to the smaller size if they differ (safety check)
    common_rows = min(rows_beacon, rows_opensnp)
    print(f"Trimming to {common_rows} common SNPs...")
    
    beacon = beacon[:common_rows, :]
    opensnp = opensnp[:common_rows, :]
    
    # 2. Create ONE permutation
    print("Generating synchronized shuffle...")
    np.random.seed(42) # Fixed seed for reproducibility
    perm = np.random.permutation(common_rows)
    
    # 3. Apply the SAME permutation to BOTH files
    beacon_shuffled = beacon[perm, :]
    opensnp_shuffled = opensnp[perm, :]
    
    # 4. Save
    print(f"Saving to {OUTPUT_BEACON} and {OUTPUT_OPENSNP}...")
    np.save(OUTPUT_BEACON, beacon_shuffled)
    np.save(OUTPUT_OPENSNP, opensnp_shuffled)
    
    print("Done! You can now run Simulate.py on these new files.")

if __name__ == "__main__":
    main()