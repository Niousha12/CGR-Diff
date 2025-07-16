import math
import os

import numpy as np
# from sklearn_extra.cluster import KMedoids

from chromosomes_holder import ChromosomesHolder, AnnotationRecord

np.random.seed(24)


class ChromosomeRepresentativeSelection:
    def __init__(self, specie, kmer, distance_metric, segment_length=None, root_path='Data'):
        self.specie = specie
        self.chromosomes_holder = ChromosomesHolder(specie, root_path=root_path)
        self.kmer = kmer
        self.length = segment_length
        self.distance_metric = distance_metric
        self.x_range = None

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.pickle_path_root = os.path.join(project_root, 'outputs', 'cache_pickles', specie)
        if not os.path.exists(self.pickle_path_root):
            os.makedirs(self.pickle_path_root)

    @staticmethod
    def find_centroid(distance_matrix, mode="mean", exclude_indices=None):
        if exclude_indices is None:
            exclude_indices = []

        # Mask the excluded indices
        mask = np.ones(distance_matrix.shape[0], dtype=bool)
        mask[exclude_indices] = False
        # Apply the mask to rows and columns
        masked_matrix = distance_matrix[mask][:, mask]

        # Initialize scores with infinities to handle fully NaN rows
        scores = np.full(masked_matrix.shape[0], np.inf)
        # Identify rows that are not fully NaN
        valid_rows = ~np.isnan(masked_matrix).all(axis=1)
        # Compute relevant scores
        if mode == "mean" or mode == "median":
            scores[valid_rows] = np.nanmean(masked_matrix[valid_rows], axis=1)  # Compute mean
            if mode == "median":
                centroid_index = np.argsort(scores)[len(scores) // 2]  # Choose index
            else:
                centroid_index = np.argmin(scores)  # Choose index
        elif mode == "medoid":
            scores[valid_rows] = np.nansum(masked_matrix[valid_rows], axis=1)
            centroid_index = np.argmin(scores)  # Choose index
        # elif mode == "kmedoid":
        #     # Replace NaNs with a large number so KMedoids can run
        #     sanitized_matrix = np.where(np.isnan(masked_matrix),
        #                                 np.nanmax(masked_matrix[np.isfinite(masked_matrix)]) * 10, masked_matrix)
        #
        #     kmedoids = KMedoids(n_clusters=1, metric='precomputed', init='random', random_state=0)
        #     kmedoids.fit(sanitized_matrix)
        #     centroid_index = kmedoids.medoid_indices_[0]
        else:
            raise ValueError("mode must be 'mean', 'median', or 'medoid'")

        # Convert the index back to the original matrix indices
        original_indices = np.arange(distance_matrix.shape[0])[mask]
        centroid_original_index = original_indices[centroid_index]

        return centroid_original_index

    @staticmethod
    def get_outliers_index_iqr(data, multiplier=1.5):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper_bound = q3 + multiplier * iqr

        outlier_indices = []
        for index, x in enumerate(data):
            if not x <= upper_bound:
                outlier_indices.append(index)
        return outlier_indices


if __name__ == '__main__':
    representative = ChromosomeRepresentativeSelection('Human', 6, 'DSSIM', segment_length=500_000)

    x_r = math.ceil(ChromosomesHolder('Human').get_largest_chromosome_length() / 500_000 / 20)
    segment_length, threshold_list, apx_list, rand_list = [], [], [], []
    for chr_name in representative.chromosomes_holder.get_all_chromosomes_name():
        representative.plot_distance_variations(chr_name, x_range=x_r, plot_random_outliers=True)
        dist_list, _, _, _ = representative.get_approximation_error(chr_name, random_outliers=False)
        threshold_list.append(np.sum(dist_list < 0.24))
        segment_length.append(len(dist_list))
        # rand_list.append(rand)

    print(sum(threshold_list) / sum(segment_length))
