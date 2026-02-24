import numpy as np
import pandas as pd
import argparse

def ped_to_binary_matrix(ped_file, max_snps=5000):

    print(f"Reading {ped_file}...")

    try:
        usecols = list(range(6 + (max_snps * 2)))
        df = pd.read_csv(ped_file, sep='\s+', header=None, usecols=usecols, dtype=str)
    except ValueError:
        print("File smaller than requested SNP count, reading entire file...")
        df = pd.read_csv(ped_file, sep='\s+', header=None, dtype=str)

    print(f"Loaded raw data. Shape: {df.shape}")
    
    genotypes = df.iloc[:, 6:].values
    n_indivs = genotypes.shape[0]
    
    print("Processing genotypes...")
    
    binary_matrix = []
    
    num_snps_to_process = min(genotypes.shape[1] // 2, max_snps)
    
    for i in range(num_snps_to_process):
        col1 = genotypes[:, 2*i]
        col2 = genotypes[:, 2*i+1]
        
        unique_alleles = np.unique(np.concatenate([col1, col2]))
        unique_alleles = unique_alleles[unique_alleles != '0']
        
        if len(unique_alleles) == 0:
            snp_vec = np.zeros(n_indivs)
        elif len(unique_alleles) == 1:
            snp_vec = np.zeros(n_indivs)
        else:
            total_alleles = np.concatenate([col1, col2])
            val, counts = np.unique(total_alleles, return_counts=True)
            sorted_indices = np.argsort(-counts)
            major_allele = val[sorted_indices[0]]
            has_minor = (col1 != major_allele) | (col2 != major_allele)
            snp_vec = has_minor.astype(int)
            
        binary_matrix.append(snp_vec)
        
    final_beacon = np.array(binary_matrix)
    
    print(f"Conversion complete. Final Beacon Shape: {final_beacon.shape} (SNPs x Individuals)")
    return final_beacon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your .ped file")
    parser.add_argument("--output", type=str, default="beacon.npy", help="Output .npy file")
    parser.add_argument("--snps", type=int, default=2000, help="Number of SNPs to extract")
    args = parser.parse_args()
    
    beacon_data = ped_to_binary_matrix(args.input, args.snps)
    np.save(args.output, beacon_data)
    print(f"Saved to {args.output}")