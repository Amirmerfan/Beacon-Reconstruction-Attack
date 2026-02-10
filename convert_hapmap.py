import numpy as np
import pandas as pd
import argparse

def ped_to_binary_matrix(ped_file, max_snps=5000):
    """
    Parses a PLINK .ped file and converts it to a binary matrix.
    
    HapMap .ped format:
    Cols 1-6: FamilyID, IndivID, Paternal, Maternal, Sex, Phenotype
    Cols 7+: Genotypes (e.g., 'A A', 'G T')
    
    Output Format:
    Rows = SNPs
    Cols = Individuals
    0 = Does not have minor allele
    1 = Has minor allele (Heterozygous or Homozygous Minor)
    """
    print(f"Reading {ped_file}...")
    
    # 1. Read the file (Skipping first 6 metadata columns)
    # We read only a subset of columns to save memory if the file is huge
    # Assuming standard whitespace separation
    try:
        # Read first 6 cols + (max_snps * 2) allele columns
        usecols = list(range(6 + (max_snps * 2)))
        df = pd.read_csv(ped_file, sep='\s+', header=None, usecols=usecols, dtype=str)
    except ValueError:
        # Fallback if file is smaller than max_snps
        print("File smaller than requested SNP count, reading entire file...")
        df = pd.read_csv(ped_file, sep='\s+', header=None, dtype=str)

    print(f"Loaded raw data. Shape: {df.shape}")
    
    # Drop the first 6 metadata columns (FamilyID, etc.)
    genotypes = df.iloc[:, 6:].values
    n_indivs = genotypes.shape[0]
    
    # The genotypes are in columns like [A, A, G, T, ...]
    # We need to combine pairs: (Col 0 + Col 1) = SNP 1
    print("Processing genotypes...")
    
    binary_matrix = []
    
    # Iterate through columns in steps of 2 (each SNP has 2 alleles)
    # limit to max_snps
    num_snps_to_process = min(genotypes.shape[1] // 2, max_snps)
    
    for i in range(num_snps_to_process):
        col1 = genotypes[:, 2*i]
        col2 = genotypes[:, 2*i+1]
        
        # Determine alleles in this column (e.g., 'A', 'G')
        # Filter out '0' which usually means missing in PLINK
        unique_alleles = np.unique(np.concatenate([col1, col2]))
        unique_alleles = unique_alleles[unique_alleles != '0']
        
        if len(unique_alleles) == 0:
            # All missing
            snp_vec = np.zeros(n_indivs)
        elif len(unique_alleles) == 1:
            # Monomorphic (only one allele exists) -> All 0s
            snp_vec = np.zeros(n_indivs)
        else:
            # Find Major vs Minor allele
            # Count occurrences
            total_alleles = np.concatenate([col1, col2])
            val, counts = np.unique(total_alleles, return_counts=True)
            # Sort by count (descending) -> Major is index 0
            sorted_indices = np.argsort(-counts)
            major_allele = val[sorted_indices[0]]
            
            # Binary Encoding:
            # 0 = Homozygous Major (Major/Major)
            # 1 = Carrying Minor (Minor/Major or Minor/Minor)
            
            # Check if individual has ANY allele that is NOT the major one
            has_minor = (col1 != major_allele) | (col2 != major_allele)
            snp_vec = has_minor.astype(int)
            
        binary_matrix.append(snp_vec)

    # Convert to numpy array
    # Current shape: (n_snps, n_indivs) -> This matches what the code expects!
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