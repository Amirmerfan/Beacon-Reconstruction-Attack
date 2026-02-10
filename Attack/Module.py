import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import hamming

class FastReconstructionTool:
    def __init__(self, beacon_shape, correlation_matrix, initial_guess=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_snps, self.num_indiv = beacon_shape
        
        self.C_target = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)
        
        if initial_guess is not None:
            guess_tensor = torch.tensor(initial_guess, dtype=torch.float32, device=self.device)
            init_logits = (guess_tensor * 4) - 2 
        else:
            init_logits = torch.randn(self.num_snps, self.num_indiv, device=self.device)

        self.B_logits = nn.Parameter(init_logits.requires_grad_(True))
        
        self.opt_corr = torch.optim.Adam([self.B_logits], lr=0.005) 
        self.opt_freq = torch.optim.Adam([self.B_logits], lr=0.05)
        
        self.sched_corr = torch.optim.lr_scheduler.ExponentialLR(self.opt_corr, gamma=0.99)
        self.sched_freq = torch.optim.lr_scheduler.ExponentialLR(self.opt_freq, gamma=0.99)

    def step_schedulers(self):
        self.sched_corr.step()
        self.sched_freq.step()

    def calculate_current_correlations(self, B_soft):
        """
        Calculates Pearson Correlation Coefficient in a differentiable way.
        """
        mean = B_soft.mean(dim=1, keepdim=True)
        B_centered = B_soft - mean
        
        cov = torch.matmul(B_centered, B_centered.T)
        
        var = cov.diag()
        std = torch.sqrt(var + 1e-8)
        
        std_matrix = torch.ger(std, std)
        corr = cov / (std_matrix + 1e-8)
        
        return corr

    def optimize(self, target_frequencies, cycles=5, corr_steps=5, freq_steps=10):
        if not isinstance(target_frequencies, torch.Tensor):
            target_freqs = torch.tensor(target_frequencies, dtype=torch.float32, device=self.device)
        else:
            target_freqs = target_frequencies.to(self.device)

        for _ in range(cycles):
            
            for _ in range(corr_steps):
                self.opt_corr.zero_grad()
                B_soft = torch.sigmoid(self.B_logits)
                
                current_corr = self.calculate_current_correlations(B_soft)
                loss_corr = torch.norm(current_corr - self.C_target, p='fro')
                
                loss_corr.backward()
                self.opt_corr.step()

            for _ in range(freq_steps):
                self.opt_freq.zero_grad()
                B_soft = torch.sigmoid(self.B_logits)
                freq_estimated = torch.sum(B_soft, dim=1)
                loss_freq = nn.functional.mse_loss(freq_estimated, target_freqs)
                loss_freq.backward()
                self.opt_freq.step()
            
        return loss_corr.item(), loss_freq.item()

    def get_reconstruction(self):
        with torch.no_grad():
            return (torch.sigmoid(self.B_logits) > 0.5).int().cpu().numpy()

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