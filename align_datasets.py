import pandas as pd
import numpy as np
import pickle
import os

def generate_aligned_datasets(data_dir, num_snps=1000, beacon_size=50, corr_size=100, seed=42):
    print(f"1. Loading top {num_snps} SNPs from Beacon_164.txt...")
    
    beacon_path = os.path.join(data_dir, "beacon_1000snps.npy")
    beacon_matrix = np.load(beacon_path)

    beacon_matrix = beacon_matrix[:num_snps, :]
    
    ref_path = os.path.join(data_dir, "reference.pickle")
    with open(ref_path, "rb") as fp:
        reference = pickle.load(fp)

    if len(reference) < num_snps:
        raise ValueError("Reference length is smaller than num_snps.")
        
    reference = np.array(reference)[:num_snps].reshape(-1, 1)
    
    binary_matrix = np.logical_and(beacon_matrix != reference, beacon_matrix != "NN").astype(np.float32)
    
    total_individuals = binary_matrix.shape[1]
    
    if beacon_size + corr_size > total_individuals:
        raise ValueError(f"Requested {beacon_size} + {corr_size} individuals, but only {total_individuals} are available.")
        
    print("2. Splitting dataset into strict Target and Correlation sets...")
    
    np.random.seed(seed)
    indices = np.random.permutation(total_individuals)
    
    beacon_idx = indices[:beacon_size]
    corr_idx = indices[beacon_size:beacon_size + corr_size]

    beacon_data = binary_matrix[:, beacon_idx]
    corr_data = binary_matrix[:, corr_idx]
    
    beacon_filename = f"target_beacon_{num_snps}.npy"
    corr_filename = f"proxy_corr_{num_snps}.npy"
    
    np.save(beacon_filename, beacon_data)
    np.save(corr_filename, corr_data)
    
    print(f"✅ Saved Target Beacon: {beacon_filename} (Shape: {beacon_data.shape})")
    print(f"✅ Saved Proxy Correlation: {corr_filename} (Shape: {corr_data.shape})")
    
    beacon_maf = np.mean(beacon_data)
    print(f"   -> Diagnostic: Target Beacon Global MAF is {beacon_maf:.4f}")

if __name__ == "__main__":
    DATA_DIRECTORY = "./" 
    generate_aligned_datasets(DATA_DIRECTORY, num_snps=1000, beacon_size=50, corr_size=100, seed=42)