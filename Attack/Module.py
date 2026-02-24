import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import hamming
from scipy.optimize import linear_sum_assignment

class ReconstructionTool(nn.Module):
    def __init__(self, beacon_shape, correlation_matrix, initial_guess=None, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_snps, self.num_indiv = beacon_shape
        
        self.C_target = torch.tensor(correlation_matrix, dtype=torch.float32, device=self.device)
        
        if initial_guess is not None:
            guess_tensor = torch.tensor(initial_guess, dtype=torch.float32, device=self.device)
            init_logits = (guess_tensor * 4) - 2 
        else:
            init_logits = torch.randn(self.num_snps, self.num_indiv, device=self.device)

        self.B_logits = nn.Parameter(init_logits.requires_grad_(True))

        self.opt_corr = torch.optim.Adam([self.B_logits], lr=0.001, fused=True) 
        self.opt_freq = torch.optim.Adam([self.B_logits], lr=0.01, fused=True)

    def calculate_current_correlations(self, B_soft):
        p11 = torch.matmul(B_soft, B_soft.T)
        row_sums = torch.sum(B_soft, dim=1, keepdim=True)
        total_matches = (2 * p11) + self.num_indiv - row_sums - row_sums.T
        return total_matches / self.num_indiv

    def _train_corr_step(self):
        self.opt_corr.zero_grad(set_to_none=True)
        self.B_logits.data.clamp_(-5.0, 5.0) 
        B_soft = torch.sigmoid(self.B_logits)
        current_corr = self.calculate_current_correlations(B_soft)
        loss = torch.norm(current_corr - self.C_target, p='fro')
        loss.backward()
        self.opt_corr.step()
        return loss

    def _train_freq_step(self, target_freqs):
        self.opt_freq.zero_grad(set_to_none=True)
        self.B_logits.data.clamp_(-5.0, 5.0)
        B_soft = torch.sigmoid(self.B_logits)
        freq_estimated = torch.mean(B_soft, dim=1)
        loss = nn.functional.mse_loss(freq_estimated, target_freqs)
        loss.backward()
        self.opt_freq.step()
        return loss

    def optimize_vg(self, target_frequencies, cycles=1, corr_steps=1000, freq_steps=500):
        if not torch.is_tensor(target_frequencies):
            target_freqs = torch.tensor(target_frequencies, dtype=torch.float32, device=self.device)
        else:
            target_freqs = target_frequencies.to(self.device)

        final_corr_loss = 0.0
        final_freq_loss = 0.0

        for _ in range(cycles):
            for _ in range(corr_steps):
                loss_tensor_corr = self._train_corr_step()
            
            final_corr_loss = loss_tensor_corr.item()
            
            for _ in range(freq_steps):
                loss_tensor_freq = self._train_freq_step(target_freqs)

            final_freq_loss = loss_tensor_freq.item()

        return final_corr_loss, final_freq_loss
    
    def optimize(self, target_frequencies, cycles=1, corr_steps=1000, freq_steps=500):
        if not torch.is_tensor(target_frequencies):
            target_freqs = torch.tensor(target_frequencies, dtype=torch.float32, device=self.device)
        else:
            target_freqs = target_frequencies.to(self.device)

        final_corr_loss = 0.0
        final_freq_loss = 0.0

        for _ in range(cycles):
            if corr_steps > 0:
                for _ in range(corr_steps):
                    loss_tensor_corr = self._train_corr_step()
                final_corr_loss = loss_tensor_corr.item()
            
            if freq_steps > 0:
                for _ in range(freq_steps):
                    loss_tensor_freq = self._train_freq_step(target_freqs)
                final_freq_loss = loss_tensor_freq.item()

        return final_corr_loss, final_freq_loss
    
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
        cost_matrix = np.zeros((num_columns, num_columns))
        
        for i in range(num_columns):
            for j in range(num_columns):
                cost_matrix[i, j] = hamming(beacon[:, i], reconstructed_beacon[:, j])
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        sorted_beacon = np.zeros_like(beacon)
        for i in range(num_columns):
            sorted_beacon[:, i] = reconstructed_beacon[:, col_ind[i]]
            
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