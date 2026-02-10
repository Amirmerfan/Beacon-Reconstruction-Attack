import numpy as np
import argparse

def shuffle_datasets(beacon_path, opensnp_path):
    print(f"Loading {beacon_path} and {opensnp_path}...")
    
    try:
        beacon = np.load(beacon_path)
        opensnp = np.load(opensnp_path)
    except FileNotFoundError:
        print("Error: Files not found. Make sure you generated Real_Beacon_Aligned.npy first.")
        return

    print(f"Original Shapes -> Beacon (HapMap): {beacon.shape}, OpenSNP: {opensnp.shape}")

    perm_beacon = np.random.permutation(beacon.shape[1])
    beacon_shuffled = beacon[:, perm_beacon]
    print(f"  -> Shuffled {beacon.shape[1]} HapMap individuals.")

    perm_opensnp = np.random.permutation(opensnp.shape[1])
    opensnp_shuffled = opensnp[:, perm_opensnp]
    print(f"  -> Shuffled {opensnp.shape[1]} OpenSNP individuals.")
    
    np.save("Shuffled_Beacon.npy", beacon_shuffled)
    np.save("Shuffled_OpenSNP.npy", opensnp_shuffled)
    
    print("-" * 30)
    print("New datasets created!")
    print("1. Shuffled_Beacon.npy")
    print("2. Shuffled_OpenSNP.npy")
    print("Run Simulate.py on these files to test a new random group.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beacon", default="Real_Beacon_Aligned.npy", help="Path to aligned beacon file")
    parser.add_argument("--opensnp", default="Real_OpenSNP_Aligned.npy", help="Path to aligned OpenSNP file")
    args = parser.parse_args()
    
    shuffle_datasets(args.beacon, args.opensnp)