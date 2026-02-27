import numpy as np
import argparse
from bed_reader import open_bed

def bed_to_binary_matrix(bed_file, max_snps=5000):

    print(f"Reading {bed_file}...")

    with open_bed(bed_file) as bed:
        n_indivs = bed.iid_count
        total_snps = bed.sid_count
        print(f"Loaded raw data. Shape: {n_indivs} individuals x {total_snps} SNPs")
        
        num_snps_to_process = min(total_snps, max_snps)
        print(f"Processing first {num_snps_to_process} SNPs...")
        
        genotypes = bed.read(index=np.s_[:, :num_snps_to_process], dtype=np.float32)
        
        genotypes = genotypes.T

        genotypes = np.nan_to_num(genotypes, nan=0.0)
        
        final_beacon = np.where(genotypes > 0, 1, 0).astype(int)

    print(f"Conversion complete. Final Beacon Shape: {final_beacon.shape} (SNPs x Individuals)")
    return final_beacon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to your .bed file")
    parser.add_argument("--output", type=str, default="beacon.npy", help="Output .npy file")
    parser.add_argument("--snps", type=int, default=2000, help="Number of SNPs to extract")
    args = parser.parse_args()
    
    beacon_data = bed_to_binary_matrix(args.input, args.snps)
    np.save(args.output, beacon_data)
    print(f"Saved to {args.output}")