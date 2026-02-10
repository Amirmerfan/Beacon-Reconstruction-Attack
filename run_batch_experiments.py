import subprocess
import re
import numpy as np
import time
import sys
import os

NUM_RUNS = 100
SHUFFLE_CMD = [sys.executable, "shuffle_data.py"]
ATTACK_CMD = [
    sys.executable, 
    os.path.join("Attack", "Simulate.py"),
    "--beacon_size", "50",
    "--snp_count", "1000",
    "--corr_epoch", "100",
    "--freq_epoch", "100",
    "--path", "Shuffled_Beacon.npy",
    "--corr_path", "Shuffled_OpenSNP.npy"
]

def run_command(command):
    """Runs a shell command and returns the stdout."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(result.stderr)
        return None
    return result.stdout

def parse_results(output):
    """Extracts Accuracy and F1 Score from Simulate.py output."""
    acc_match = re.search(r"Accuracy:\s+([0-9.]+)", output)
    f1_match = re.search(r"F1 Score:\s+([0-9.]+)", output)
    
    if acc_match and f1_match:
        return float(acc_match.group(1)), float(f1_match.group(1))
    return None, None

def main():
    accuracies = []
    f1_scores = []
    
    print(f"Starting batch of {NUM_RUNS} experiments...")
    print("-" * 60)
    print(f"{'Run':<5} | {'Accuracy':<10} | {'F1 Score':<10} | {'Status':<10}")
    print("-" * 60)

    start_time = time.time()

    for i in range(1, NUM_RUNS + 1):
        run_command(SHUFFLE_CMD)
        
        output = run_command(ATTACK_CMD)
        
        if output:
            acc, f1 = parse_results(output)
            
            if acc is not None:
                accuracies.append(acc)
                f1_scores.append(f1)
                print(f"{i:<5} | {acc:.4f}     | {f1:.4f}     | Done")
            else:
                print(f"{i:<5} | {'N/A':<10} | {'N/A':<10} | Failed to Parse")
        else:
            print(f"{i:<5} | {'N/A':<10} | {'N/A':<10} | Error")

    total_time = time.time() - start_time
    
    print("-" * 60)
    print("FINAL RESULTS")
    print("-" * 60)
    
    if len(accuracies) > 0:
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"Total Runs:       {len(accuracies)}")
        print(f"Total Time:       {total_time:.2f} seconds ({total_time/len(accuracies):.2f} s/run)")
        print(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}  ± {std_f1:.4f}")
        print("-" * 60)
        
        print("INTERPRETATION:")
        print(f"The attack reconstructs {avg_acc*100:.1f}% of the genome correctly on average.")
        print(f"The F1 score of {avg_f1:.3f} indicates the balance between Precision and Recall.")
        if std_f1 > 0.05:
            print("(!) High Variance: Results depend heavily on WHICH people are in the group.")
        else:
            print("(OK) Low Variance: Attack is consistent across different victim groups.")
            
    else:
        print("No successful runs recorded.")

if __name__ == "__main__":
    main()