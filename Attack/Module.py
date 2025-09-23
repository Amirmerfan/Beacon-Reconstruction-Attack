import numpy as np
import math
import torch
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt

class ReconstructionTool:
    def __init__(self, beacon):
        self.beacon = beacon

    def query(self, proportion=1.0):
        """
        Generates a reconstructed beacon based on the proportion of 1's to retain.
        
        Args:
            proportion (float): Proportion of 1's to query in each row. Default is 1.0.

        Returns:
            np.ndarray: Reconstructed beacon matrix.
        """
        # Initialize 
        reconstructed_beacon = np.zeros_like(self.beacon)

        # Iterate 
        for idx, row in enumerate(self.beacon):
            # Calculate the number of 1's to assign based on the given proportion
            num_ones = int(np.sum(row) * proportion)

            # Generate random 
            indices_to_fill = np.random.choice(self.beacon.shape[1], size=num_ones, replace=False)

            # Assign 1's 
            reconstructed_beacon[idx, indices_to_fill] = 1

        return reconstructed_beacon  # Return the reconstructed beacon

    @staticmethod
    def visualize_beacon(beacon):
        """
        Visualizes the beacon matrix as a heatmap.
        """
        plt.imshow(beacon, cmap='gray')
        plt.xlabel('Individuals')
        plt.ylabel('SNPs')
        plt.title('Beacon')
        plt.show()

    @staticmethod
    def calculate_correlations(beacon):
        '''
        Calculates the correlations between each pair of SNPs based on the Sokal-Michener similarity.

        Parameters:
            beacon: A numpy matrix with 0's and 1's representing SNP data.

        Returns:   
            A numpy matrix with the correlations.
        '''
        #population
        N_p = beacon.shape[1]

        #  pairs of SNPs
        correlations = beacon @ beacon.T / N_p

        return correlations


    @staticmethod
    def calculate_accuracy(beacon, new_beacon):
        """
        Calculates accuracy between the original and reconstructed beacons.
        """
        true_positives = np.sum((beacon == 1) & (new_beacon == 1))
        true_negatives = np.sum((beacon == 0) & (new_beacon == 0))
        false_positives = np.sum((beacon == 0) & (new_beacon == 1))
        false_negatives = np.sum((beacon == 1) & (new_beacon == 0))
        return (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

    @staticmethod
    def calculate_precision(beacon, new_beacon):
        """
        Calculates precision between the original and reconstructed beacons.
        """
        true_positives = np.sum((beacon == 1) & (new_beacon == 1))
        false_positives = np.sum((beacon == 0) & (new_beacon == 1))
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def calculate_recall(beacon, new_beacon):
        """
        Calculates recall between the original and reconstructed beacons.
        """
        true_positives = np.sum((beacon == 1) & (new_beacon == 1))
        false_negatives = np.sum((beacon == 1) & (new_beacon == 0))
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_f1(beacon, new_beacon):
        """
        Calculates F1 score between the original and reconstructed beacons.
        """
        precision = ReconstructionTool.calculate_precision(beacon, new_beacon)
        recall = ReconstructionTool.calculate_recall(beacon, new_beacon)
        return 2 * ((precision * recall) / (precision + recall))

    @staticmethod
    def frequency_loss(reconstructed_beacon, target_frequencies):
        """
        Calculates frequency loss for the reconstructed beacon.

        Args:
            reconstructed_beacon (torch.Tensor): Reconstructed beacon matrix.
            target_frequencies (torch.Tensor): Target frequencies for each row.

        Returns:
            torch.Tensor: Frequency loss value.
        """
        current_frequencies = torch.sum(reconstructed_beacon, dim=1)
        loss = torch.mean((current_frequencies - target_frequencies) ** 2)
        return loss

    @staticmethod
    def compare_and_sort_columns(beacon, reconstructed_beacon):
        """
        Aligns columns of the reconstructed beacon to match the original beacon.

        Args:
            beacon (np.ndarray): Original beacon matrix.
            reconstructed_beacon (np.ndarray): Reconstructed beacon matrix.

        Returns:
            np.ndarray: Column-aligned reconstructed beacon.
        """
        num_columns_beacon = beacon.shape[1]
        reconstructed_beacon1 = np.zeros_like(beacon)

        for i in range(num_columns_beacon):
            hamming_distances = [
                (hamming(beacon[:, i].flatten(), reconstructed_beacon[:, j].flatten()), j)
                for j in range(reconstructed_beacon.shape[1])
            ]
            _, most_similar_column = min(hamming_distances)
            reconstructed_beacon1[:, i] = reconstructed_beacon[:, most_similar_column]

        return reconstructed_beacon1

    @staticmethod
    def print_metrics_table(individuals, accuracies, precisions, recalls, f1_scores):
        """
        Prints the evaluation metrics in a tabular format.
        """
        print("\nResults Summary:")
        print("Individuals\tAccuracy\tPrecision\tRecall\t\tF1 Score")
        for i in range(len(individuals)):
            print(f"{individuals[i]}\t\t{accuracies[i]:.3f}\t\t{precisions[i]:.3f}\t\t{recalls[i]:.3f}\t\t{f1_scores[i]:.3f}")