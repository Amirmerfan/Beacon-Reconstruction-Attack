import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import hamming

class FastReconstructionTool:
    def __init__(self, beacon_shape, correlation_matrix, initial_guess=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_snps, self.num_indiv = beacon_shape
        
        # Ensure Correlation Matrix is on the correct device
        self.C_target = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)
        
        # Initialize Logits
        if initial_guess is not None:
            guess_tensor = torch.tensor(initial_guess, dtype=torch.float32, device=self.device)
            # Map 0 -> -2, 1 -> +2
            init_logits = (guess_tensor * 4) - 2 
        else:
            init_logits = torch.randn(self.num_snps, self.num_indiv, device=self.device)

        self.B_logits = nn.Parameter(init_logits.requires_grad_(True))
        
        # --- FIX: EQUAL LEARNING RATES (Paper Standard) ---
        # Both set to 0.001. Previous 0.01 was too aggressive.
        self.opt_corr = torch.optim.Adam([self.B_logits], lr=0.001) 
        self.opt_freq = torch.optim.Adam([self.B_logits], lr=0.01) 

        # --- SCHEDULERS ---
        self.sched_corr = torch.optim.lr_scheduler.ExponentialLR(self.opt_corr, gamma=0.96)
        self.sched_freq = torch.optim.lr_scheduler.ExponentialLR(self.opt_freq, gamma=0.96)

    def calculate_current_correlations(self, B_soft):
        if B_soft.device != self.device:
            B_soft = B_soft.to(self.device)
            
        p11 = torch.matmul(B_soft, B_soft.T)
        B_neg = 1.0 - B_soft
        p00 = torch.matmul(B_neg, B_neg.T)

        sm_matrix = (p11 + p00) / self.num_indiv
        return sm_matrix

    def optimize(self, target_frequencies, cycles=1, corr_steps=1000, freq_steps=500):
        if not isinstance(target_frequencies, torch.Tensor):
            target_freqs = torch.tensor(target_frequencies, dtype=torch.float32, device=self.device)
        else:
            target_freqs = target_frequencies.to(self.device)

        loss_corr_val = 0
        loss_freq_val = 0

        for _ in range(cycles):
            
            # --- PHASE 1: CORRELATION ---
            for _ in range(corr_steps):
                self.opt_corr.zero_grad()
                
                # Clamp to [-10, 10]
                self.B_logits.data.clamp_(-10.0, 10.0)
                
                B_soft = torch.sigmoid(self.B_logits)
                current_corr = self.calculate_current_correlations(B_soft)
                
                loss_corr = torch.norm(current_corr - self.C_target, p='fro')
                loss_corr.backward()
                self.opt_corr.step()
                loss_corr_val = loss_corr.item()

            # --- PHASE 2: FREQUENCY ---
            for _ in range(freq_steps):
                self.opt_freq.zero_grad()
                
                # Clamp to [-10, 10]
                self.B_logits.data.clamp_(-10.0, 10.0)
                
                B_soft = torch.sigmoid(self.B_logits)
                freq_estimated = torch.sum(B_soft, dim=1)
                
                loss_freq = nn.functional.mse_loss(freq_estimated, target_freqs)
                loss_freq.backward()
                self.opt_freq.step()
                loss_freq_val = loss_freq.item()
            
            # Step Schedulers
            self.sched_corr.step()
            self.sched_freq.step()
            
        return loss_corr_val, loss_freq_val

    def step_schedulers(self):
        pass

    def get_reconstruction(self):
        with torch.no_grad():
            return (torch.sigmoid(self.B_logits) > 0.5).int().cpu().numpy()

    # --- Metrics & Helpers ---
    @staticmethod
    def calculate_accuracy(beacon, new_beacon):
        return np.mean(beacon == new_beacon)
        
    @staticmethod
    def calculate_f1(beacon, new_beacon):
        tp = np.sum((beacon == 1) & (new_beacon == 1))
        fp = np.sum((beacon == 0) & (new_beacon == 1))
        fn = np.sum((beacon == 1) & (new_beacon == 0))
        if tp == 0: return 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def compare_and_sort_columns(beacon, reconstructed_beacon):     
        num_columns = beacon.shape[1]
        sorted_beacon = np.zeros_like(beacon)
        used_indices = set()
        for i in range(num_columns):
            best_j = -1
            min_dist = float('inf')
            for j in range(num_columns):
                if j in used_indices: continue
                dist = hamming(beacon[:, i], reconstructed_beacon[:, j])
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
            sorted_beacon[:, i] = reconstructed_beacon[:, best_j]
            used_indices.add(best_j)
        return sorted_beacon
    
    @staticmethod
    def greedy_initialization(beacon_freqs, N):
        M = len(beacon_freqs)
        guess = np.zeros((M, N))
        for i in range(M):
            count = int(beacon_freqs[i]) if beacon_freqs[i] > 1 else int(round(beacon_freqs[i] * N))
            indices = np.random.choice(N, count, replace=False)
            guess[i, indices] = 1.0
        return guess