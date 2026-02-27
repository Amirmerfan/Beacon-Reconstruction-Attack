# Beacon Reconstruction Attack
>This repository contains the implementation of a novel beacon reconstruction attack that exploits genomic data-sharing beacons by using single nucleotide polymorphism (SNP) correlations and summary statistics. The attack uncovers critical privacy vulnerabilities, allowing the reconstruction of genomic data for individuals in a beacon database. The study demonstrates the feasibility of such attacks and emphasizes the need for enhanced privacy-preserving mechanisms in genomic data sharing.

> **Keywords**: Genome Reconstruction Attacks, [Genomic Data Sharing Beacons](https://en.wikipedia.org/wiki/Global_Alliance_for_Genomics_and_Health), [ Genome Privacy](https://en.wikipedia.org/wiki/Genetic_privacy), [Single-nucleotide polymorphism](https://en.wikipedia.org/wiki/Single-nucleotide_polymorphism)

> ---

## Authors

Kousar Saleem, A. Ercument Cicek, and Sinem Sav  

---

## Forked and Optimized by

Amirmohammad Erfan

---

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Citations](#citations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Baseline Reconstruction Algorithm**: A greedy approach to reconstruct genomic data without accounting for SNP correlations.
- **Gradient-Based Optimization**: An advanced algorithm alternating between minimizing SNP correlation loss and allele frequency matching.
- **Flexible Configurations**: Allows customization of beacon size, SNP subset size, and attacker knowledge (e.g., leaked genomes).
- **Performance Evaluation**: Detailed analysis of reconstruction effectiveness under various scenarios.

---
## Getting Started

- Clone the repository:
   ```bash
   cd Beacon-Reconstruction-Attack
---

## Installation

- You need to install the following dependencies in Python3 for this project:
   ```bash
   pip3 install numpy scipy matplotlib torch pandas 

---

## Usage Instructions
### Running Baseline Simulations
- Use the `baseline.py` script to execute the baseline beacon reconstruction attack. Parameters allow you to configure beacon size and SNP count.
   ```bash
   python baseline.py --beacon_Size 50 --snp_count 30

### Key Arguments

- `--beacon_Size`: Number of individuals targeted for reconstruction.
- `--snp_count`: Size of the SNP subset in the beacon.
 
### Running Simulations

**1. Single Optimization Run (`simulate.py`)**
- Use the `simulate.py` script to execute a single instance of the beacon reconstruction attack. Parameters allow you to configure beacon size, SNP count, and the number of training epochs for correlation and frequency optimization.
  ```bash
  python simulate.py --beacon_size 50 --snp_count 1000 --corr_epoch 1001 --freq_epoch 501 --path Beacon.npy --corr_path OpenSNP.npy --num_runs 1

**2. Batch Optimization Run (`simulate_all.py`)**

- Use the `simulate_all.py` script to run the attack across multiple data splits automatically. It iterates through all target beacons and reference files in a specified folder, uses early stopping based on convergence, and outputs the final averaged metrics (Accuracy, F1 Score, and Time).

   ```bash
   python simulate_all.py --beacon_size 50 --snp_count 1000 --corr_epoch 1001 --freq_epoch 501 --folder ./1000snps/hapmap

### Key Arguments

### Key Arguments

* `--beacon_Size`: Number of individuals targeted for reconstruction.
* `--snp_count`: Size of the SNP subset in the beacon.
* `--corr_epoch`: Stage 1 of optimization for correlation loss.
* `--freq_epoch`: Stage 2 of optimization for frequency loss.
* `--folder`: The directory containing the batched data splits (e.g., `Target_Beacon_i.npy` and `Reference_i.npy` files) for batch execution.
* `--path`: The direct file path to a specific target beacon `.npy` file (used for single runs).
* `--corr_path`: The direct file path to the corresponding reference correlation matrix `.npy` file (used for single runs).
* `--num_runs`: The maximum number of beacon splits or iterations to process during a batch execution.

---
## License

[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)


© 2024 Beacon Defender Framework.

**For commercial use, please contact.**


---

## Acknowledgments

We would like to thank Göktuğ Gürbüztürk, Efe Erkan, Deniz Aydemir, İrem Aydın, Kerem Ayöz, and Erman Ayday for their contributions to this project.
