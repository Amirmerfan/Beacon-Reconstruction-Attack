import numpy as np
import pandas as pd
from pandas_plink import read_plink
import argparse
import os

def align_datasets(hapmap_map, opensnp_prefix, beacon_path="beacon.npy"):
    print("--- Loading Datasets ---")
    
    
    print(f"Reading HapMap Map: {hapmap_map}...")
    try:
        
        hapmap_snps = pd.read_csv(hapmap_map, sep=r'\s+', header=None, names=['Chr', 'rsID', 'Dist', 'Pos'], dtype=str)
    except Exception as e:
        print(f"Error reading map file: {e}")
        return
    
    print(f"Reading HapMap Genotypes ({beacon_path})...")
    if os.path.exists(beacon_path):
        hapmap_matrix = np.load(beacon_path)
    else:
        print(f"Error: {beacon_path} not found. Please run convert_hapmap.py first.")
        return

    if hapmap_matrix.shape[0] != len(hapmap_snps):
        print(f"Warning: SNP count mismatch! Map has {len(hapmap_snps)}, Matrix has {hapmap_matrix.shape[0]}.")
        print("Truncating to the smaller size...")
        min_len = min(len(hapmap_snps), hapmap_matrix.shape[0])
        hapmap_snps = hapmap_snps.iloc[:min_len]
        hapmap_matrix = hapmap_matrix[:min_len, :]

    
    print(f"Reading OpenSNP PLINK files with prefix: {opensnp_prefix}...")
    
    if not os.path.exists(opensnp_prefix + ".bed"):
        print(f"Error: PLINK files not found at {opensnp_prefix}.bed")
        print("Make sure you extracted the .tar.gz and are pointing to the file prefix!")
        return

    try:
        (bim, fam, bed) = read_plink(opensnp_prefix, verbose=False)
        opensnp_snps = bim['snp'].values 
        print(f"OpenSNP Loaded: {bed.shape[0]} SNPs, {bed.shape[1]} Individuals")
    except Exception as e:
        print(f"Error reading OpenSNP: {e}")
        return
    
    print("--- Aligning SNPs ---")
    common_rsids = set(hapmap_snps['rsID']).intersection(set(opensnp_snps))
    common_rsids = list(common_rsids)
    print(f"Found {len(common_rsids)} common SNPs between datasets.")
    
    if len(common_rsids) < 100:
        print("Error: Too few common SNPs! Check if your file formats are correct.")
        return

    print("Extracting HapMap subset...")
    hapmap_df = pd.DataFrame(hapmap_snps)
    hapmap_df['original_index'] = range(len(hapmap_df))
    
    hapmap_common = hapmap_df[hapmap_df['rsID'].isin(common_rsids)].sort_values('rsID')
    hapmap_indices = hapmap_common['original_index'].values
    
    aligned_beacon = hapmap_matrix[hapmap_indices, :]
    
    print("Extracting OpenSNP subset (this might take a moment)...")
    opensnp_df = pd.DataFrame({'rsID': opensnp_snps, 'original_index': range(len(opensnp_snps))})
    opensnp_common = opensnp_df[opensnp_df['rsID'].isin(common_rsids)].sort_values('rsID')
    opensnp_indices = opensnp_common['original_index'].values
    
    aligned_opensnp_raw = bed[opensnp_indices, :].compute()
    
    aligned_opensnp_raw = np.nan_to_num(aligned_opensnp_raw)
    aligned_opensnp = (aligned_opensnp_raw > 0).astype(int)
    
    print("--- Saving Aligned Data ---")
    np.save("Real_Beacon_Aligned.npy", aligned_beacon)
    np.save("Real_OpenSNP_Aligned.npy", aligned_opensnp)
    
    print("Success!")
    print(f"1. Real_Beacon_Aligned.npy  (Shape: {aligned_beacon.shape})")
    print(f"2. Real_OpenSNP_Aligned.npy (Shape: {aligned_opensnp.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hapmap_map", required=True, help="Path to HapMap .map file")
    parser.add_argument("--opensnp_prefix", required=True, help="Path to OpenSNP file prefix (without .bed)")
    parser.add_argument("--beacon_path", default="beacon.npy", help="Path to your existing beacon.npy")
    args = parser.parse_args()
    
    align_datasets(args.hapmap_map, args.opensnp_prefix, args.beacon_path)