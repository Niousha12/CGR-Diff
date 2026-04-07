import math
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from chaos_game_representation import CGR
from chromosomes_holder import ChromosomesHolder
from compress_dna import compressed_size, FastDNACompressor
from distances.distance_metrics import get_dist


class ChromosomeRepresentativeSelection:
    def __init__(self, specie, kmer, distance_metric, segment_length=None, root_path='Data'):
        self.specie = specie
        self.chromosomes_holder = ChromosomesHolder(specie, root_path=root_path)
        self.kmer = kmer
        self.length = segment_length
        self.distance_metric = distance_metric
        self.x_range = None

        self.compression_fn = FastDNACompressor(min_match=kmer, seed_len=kmer, window=self.length)
        self.compression_threshold = 20000

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.pickle_path_root = os.path.join(project_root, 'outputs', 'cache_pickles', specie)
        if not os.path.exists(self.pickle_path_root):
            os.makedirs(self.pickle_path_root)

    def get_representative_of_representatives(self, pipeline="RepSeg"):
        representative_dict_list = []
        for chromosome_name in self.chromosomes_holder.get_all_chromosomes_name():
            if pipeline == "RepSeg":
                representative_dict_list.append(self.get_representative(chromosome_name))
            elif pipeline == "aRepSeg":
                representative_dict_list.append(self.get_approximate_representative(chromosome_name))
        representative_fcgrs = [rep['fcgr'] for rep in representative_dict_list]

        distance_matrix = np.zeros((len(representative_fcgrs), len(representative_fcgrs)))
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1):
                distance_matrix[i, j] = get_dist(representative_fcgrs[i], representative_fcgrs[j],
                                                 dist_m=self.distance_metric)
                distance_matrix[j, i] = distance_matrix[i, j]

        representative_of_representatives_index = self.find_centroid(distance_matrix, exclude_indices=None)
        representative_of_representatives = representative_dict_list[representative_of_representatives_index]

        return representative_of_representatives

    def get_representative(self, chromosome_name, compress=False):
        segments = self.get_non_overlapping_segments(chromosome_name)

        exclude_indices = None
        if compress:
            exclude_indices = []
            compress_sizes = []
            for i, seq in tqdm(enumerate(segments['segments_sequences'])):
                if seq.count('N') > 0:
                    exclude_indices.append(i)
                    # compress_sizes.append(float('inf'))
                    continue
                tokens = self.compression_fn.compress(seq)
                size = compressed_size(tokens)
                compress_sizes.append(size)
                if size < self.compression_threshold:
                    exclude_indices.append(i)
            print(f"exclude_indices for chromosome {chromosome_name} are {exclude_indices}")
            # print(compress_sizes)

        fcgrs = self.get_fcgrs_of_segments(chromosome_name)
        distance_matrix = self.get_distance_matrix(chromosome_name)

        centroid = self.find_centroid(distance_matrix, exclude_indices=exclude_indices)

        return {"sequence": segments['segments_sequences'][centroid],
                "chromosome": chromosome_name,
                "index": centroid,
                "fcgr": fcgrs[centroid],
                "type": "non-approximative"}

    def get_random_representative_outlier_condition(self, chromosome_name, outlier=True):
        segments = self.get_non_overlapping_segments(chromosome_name)
        fcgrs = self.get_fcgrs_of_segments(chromosome_name)

        random_centroid_index = self.chromosomes_holder.choose_random_fragment_index(chromosome_name, self.length,
                                                                                     outlier=outlier)

        return {"sequence": segments['segments_sequences'][random_centroid_index],
                "index": random_centroid_index,
                "fcgr": fcgrs[random_centroid_index],
                "type": f"random_outlier_{outlier}"}

    def get_random_representative(self, chromosome_name):
        segments = self.get_non_overlapping_segments(chromosome_name)
        fcgrs = self.get_fcgrs_of_segments(chromosome_name)

        random_centroid_index = np.random.randint(0, len(segments['segments_sequences']))

        return {"sequence": segments['segments_sequences'][random_centroid_index],
                "index": random_centroid_index,
                "fcgr": fcgrs[random_centroid_index],
                "type": f"random"}

    def get_approximate_representative(self, chromosome_name, random_sequences_number=30,
                                       remove_outliers_function="IQR", verbose=False):
        if verbose:
            print(f"Finding the approximate representative for chromosome {chromosome_name}")
        random_sequences_list = []
        count, outlier_indices_number_total = 0, 0
        avgs = None
        while len(random_sequences_list) < random_sequences_number:
            count += 1
            for _ in range(random_sequences_number - len(random_sequences_list)):
                random_sequence_dict = self.chromosomes_holder.get_random_segment(self.length, chromosome_name,
                                                                                  return_dict=True)
                fcgr = CGR(random_sequence_dict['sequence'], self.kmer).get_fcgr()
                random_sequence_dict['fcgr'] = fcgr

                random_sequences_list.append(random_sequence_dict)

            # Get the distance matrix between these choices, then get the average distance for each choice
            distance_matrix = np.zeros((len(random_sequences_list), len(random_sequences_list)))
            for i in range(distance_matrix.shape[0]):
                for j in range(i + 1):
                    distance_matrix[i, j] = get_dist(random_sequences_list[i]['fcgr'], random_sequences_list[j]['fcgr'],
                                                     dist_m=self.distance_metric)
                    distance_matrix[j, i] = distance_matrix[i, j]
            avgs = np.mean(distance_matrix, axis=1)
            if verbose:
                print(f"Averages at count {count} are : {avgs}, "
                      f"Mean of these averages : {np.mean(avgs)}")

            # Find the outlier indices
            if remove_outliers_function == "ZSCORE":
                outlier_indices = self.get_outliers_index_zscore(avgs)
            elif remove_outliers_function == "IQR":
                outlier_indices = self.get_outliers_index_iqr(avgs)
            else:
                raise ValueError("Invalid outlier removal function")
            outlier_indices_number_total += len(outlier_indices)
            if verbose:
                print(f"indices dropped at count {count} are {outlier_indices}")

            # Remove the outliers from the list of dictionaries
            random_sequences_list = [item for idx, item in enumerate(random_sequences_list) if
                                     idx not in outlier_indices]

        # Choose the minimum average distance from the remaining as the representative
        representative_dict = random_sequences_list[np.argmin(avgs)]
        if verbose:
            print(f"The process ran for {count} times, "
                  f"Total number of dropped indices {outlier_indices_number_total}")

        return {"sequence": representative_dict['sequence'],
                "start": representative_dict['start'],
                "fcgr": representative_dict['fcgr'],
                "type": "approximative"}

    def get_distance_from_representative(self, chromosome_name, representative_dict):
        if representative_dict['type'] == "approximative":
            fcgrs = self.get_fcgrs_of_segments(chromosome_name)
            distance_from_representative = np.zeros(len(fcgrs))

            for index, fcgr in enumerate(fcgrs):
                distance_from_representative[index] = get_dist(fcgr, representative_dict['fcgr'],
                                                               dist_m=self.distance_metric)
        else:
            distance_matrix = self.get_distance_matrix(chromosome_name)
            distance_from_representative = distance_matrix[representative_dict['index'], :]

        return distance_from_representative

    def get_approximation_error(self, chromosome_name, num_samples=30, random_outliers=True):
        representative = self.get_representative(chromosome_name)
        distances_from_representative = self.get_distance_from_representative(chromosome_name, representative)

        approximative_representative = self.get_approximate_representative(chromosome_name, num_samples)
        distances_from_approximative_representative = \
            self.get_distance_from_representative(chromosome_name, approximative_representative)

        approximation_error = np.mean(
            np.abs((distances_from_approximative_representative - distances_from_representative)))

        random_error = None
        if random_outliers:
            random_representative = self.get_random_representative_outlier_condition(chromosome_name)
            distances_from_random_representative = \
                self.get_distance_from_representative(chromosome_name, random_representative)
            random_error = np.mean(
                np.abs((distances_from_random_representative - distances_from_representative)))

        return distances_from_representative, approximation_error, random_error

    def plot_distance_variations(self, chromosome_name, plot_random_outliers=True, plot_approximate=True,
                                 random_sequences_number=30, x_range=None, plot_compress=False):
        prefix = "RepSeg"
        if plot_approximate:
            prefix = "RepSeg_and_aRepSeg"
            if plot_random_outliers:
                prefix = "RepSeg_and_aRepSeg_and_random"
        elif plot_random_outliers:
            prefix = "RepSeg_and_random"

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        figure_path = os.path.join(project_root, 'Figures', 'Representative', self.specie, prefix)
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        plt.figure(figsize=(10, 5))

        pipeline_representative_dict = self.get_representative(chromosome_name)
        distances_from_centroid = self.get_distance_from_representative(chromosome_name, pipeline_representative_dict)

        if plot_compress:
            compress_pipeline_representative_dict = self.get_representative(chromosome_name, compress=True)
            distances_from_compress_centroid = self.get_distance_from_representative(chromosome_name,
                                                                                     compress_pipeline_representative_dict)
            plt.plot(distances_from_compress_centroid, marker='o', linestyle='-', markersize=4, color='blue')

        if plot_random_outliers:
            random_outlier_representative_dict = self.get_random_representative_outlier_condition(chromosome_name, True)
            distance_from_outlier_representative = self.get_distance_from_representative(chromosome_name,
                                                                                         random_outlier_representative_dict)
            plt.plot(distance_from_outlier_representative, marker='o', linestyle='-', markersize=4, color='black')

        if plot_approximate:
            approximate_representative_dict = self.get_approximate_representative(chromosome_name,
                                                                                  random_sequences_number)
            distance_from_approximate_representative = self.get_distance_from_representative(chromosome_name,
                                                                                             approximate_representative_dict)
            plt.plot(distance_from_approximate_representative, marker='o', linestyle='-', markersize=4, color='blue')

        plt.plot(distances_from_centroid, marker='o', linestyle='-', markersize=4, color='red')
        plt.grid(True)

        if chromosome_name == "Whole Genome":
            include_whole_genome = True
        else:
            include_whole_genome = False

        if self.x_range is None:
            if x_range is None:
                self.x_range = math.ceil(
                    self.chromosomes_holder.get_largest_chromosome_length(include_whole_genome) / self.length / 20)
            else:
                self.x_range = x_range

        x_ticks = []
        for i in range(0, int(self.x_range) + 1):
            x_ticks.append(i * 20)
        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.tight_layout()
        plt.savefig(f'{figure_path}/chr-{chromosome_name}_kmer-{self.kmer}_length-{self.length}-compress.png')
        # plt.show()
        plt.close()

    def get_non_overlapping_segments(self, chromosome_name):
        pickle_path = os.path.join(self.pickle_path_root, f"chr_{chromosome_name}_len_{self.length}_segments.pickle")
        if os.path.exists(pickle_path):
            return self.load_pickle(pickle_path)
        else:
            segments = self.chromosomes_holder.get_chromosome_non_overlapping_segments(chromosome_name, self.length)
            self.create_pickle(segments, pickle_path)
            return segments

    def get_fcgrs_of_segments(self, chromosome_name):
        pickle_path = os.path.join(self.pickle_path_root,
                                   f"chr_{chromosome_name}_len_{self.length}_kmer_{self.kmer}_fcgrs.pickle")
        if os.path.exists(pickle_path):
            return self.load_pickle(pickle_path)
        else:
            segments = self.get_non_overlapping_segments(chromosome_name)

            fcgrs_list = []
            for segment in tqdm(segments['segments_sequences']):
                fcgrs_list.append(CGR(segment, self.kmer).get_fcgr())
            self.create_pickle(fcgrs_list, pickle_path)
            return fcgrs_list

    def get_distance_matrix(self, chromosome_name):
        pickle_path = os.path.join(self.pickle_path_root,
                                   f"chr_{chromosome_name}_len_{self.length}_kmer_{self.kmer}"
                                   f"_dist_{self.distance_metric}_distance_matrix.pickle")
        if os.path.exists(pickle_path):
            return self.load_pickle(pickle_path)
        else:
            fcgrs = self.get_fcgrs_of_segments(chromosome_name)

            distance_matrix = np.zeros((len(fcgrs), len(fcgrs)))
            for i in tqdm(range(distance_matrix.shape[0])):
                for j in range(i + 1):
                    distance_matrix[i, j] = get_dist(fcgrs[i], fcgrs[j], dist_m=self.distance_metric)
                    distance_matrix[j, i] = distance_matrix[i, j]
            self.create_pickle(distance_matrix, pickle_path)
            return distance_matrix

    @staticmethod
    def create_pickle(results, pickle_path):
        if not os.path.exists(pickle_path):
            with open(pickle_path, 'wb') as handle:
                pickle.dump(results, handle)
        else:
            print("Pickle file already exists")

    @staticmethod
    def load_pickle(pickle_path):
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as handle:
                results = pickle.load(handle)
            return results
        else:
            raise FileNotFoundError("Pickle file not found")

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
    specie = 'Bombina bombina'
    representative = ChromosomeRepresentativeSelection(specie, 6, 'DSSIM', segment_length=500_000)

    # segment_length, threshold_list, apx_list, rand_list = [], [], [], []
    for chr_name in representative.chromosomes_holder.get_all_chromosomes_name():
        print(f"Length of chromosome {chr_name} is "
              f"{len(representative.chromosomes_holder.get_chromosome_sequence(chr_name))}")
        # x_range = math.ceil(len(representative.chromosomes_holder.get_chromosome_sequence(chr_name)) / 500000 / 20)
        representative.plot_distance_variations(chr_name, x_range=None, plot_random_outliers=False,
                                                plot_approximate=False, plot_compress=False)
        # dist_list, _, _ = representative.get_approximation_error(chr_name, random_outliers=False)
        # threshold_list.append(np.sum(dist_list < 0.24))
        # segment_length.append(len(dist_list))
    # print(sum(threshold_list) / sum(segment_length))

    # # Run an analysis on some lists
    # list_chr_Y = [41945, 49184, 46501, 50703, 54145, 58752, 59090, 59020, 58642, 57859, 58954, 58884, 56932, 57749, 58476, 57242, 55552, 54265, 5038, 21674, 55863, 17213, 49123, 32422, 7847, 50191, 57366, 57844, 57811, 57167, 58799, 58009, 58509, 57566, 58538, 57056, 58597, 57425, 58569, 58631, 57582, 55018, 39326, 56894, 54800, 55525, 57286, 56418, 55072, 57925, 58696, 52972, 58134, 58236, 54378, 7296, 3883, 7139, 8611, 6063, 4881, 5759, 6047, 5900, 5514, 5577, 6839, 5676, 2861, 5518, 4845, 6901, 6816, 6445, 7298, 5708, 6875, 6256, 6735, 7073, 6127, 4773, 4830, 5831, 3647, 6047, 6594, 6183, 6781, 6123, 5989, 5465, 3933, 6036, 4045, 5794, 6848, 6217, 6033, 6866, 6484, 4973, 5403, 5557, 5430, 5657, 5577, 5816, 5542, 5767, 5519, 5349, 6469, 4853, 5112, 4689, 2963, 5720, 5181, 5533, 4578, 6638, 8944]
    # y = np.asarray(list_chr_Y)

'''
Chromosome Y:
[41945, 49184, 46501, 50703, 54145, 58752, 59090, 59020, 58642, 57859, 58954, 58884, 56932, 57749, 58476, 57242, 55552, 54265, 5038, 21674, 55863, 17213, 49123, 32422, 7847, 50191, 57366, 57844, 57811, 57167, 58799, 58009, 58509, 57566, 58538, 57056, 58597, 57425, 58569, 58631, 57582, 55018, 39326, 56894, 54800, 55525, 57286, 56418, 55072, 57925, 58696, 52972, 58134, 58236, 54378, 7296, 3883, 7139, 8611, 6063, 4881, 5759, 6047, 5900, 5514, 5577, 6839, 5676, 2861, 5518, 4845, 6901, 6816, 6445, 7298, 5708, 6875, 6256, 6735, 7073, 6127, 4773, 4830, 5831, 3647, 6047, 6594, 6183, 6781, 6123, 5989, 5465, 3933, 6036, 4045, 5794, 6848, 6217, 6033, 6866, 6484, 4973, 5403, 5557, 5430, 5657, 5577, 5816, 5542, 5767, 5519, 5349, 6469, 4853, 5112, 4689, 2963, 5720, 5181, 5533, 4578, 6638, 8944]
Chromosome 21:
[38929, 12091, 18453, 11194, 49152, 54288, 19581, 6908, 6356, 6422, 6358, 44896, 40019, 39457, 40133, 49958, 38762, 47843, 50771, 57658, 36421, 40251, 22845, 58138, 57386, 59072, 59095, 59603, 58866, 58884, 58587, 56166, 58616, 59118, 59189, 59171, 58928, 58848, 59088, 59231, 59202, 58928, 58731, 59018, 58698, 59004, 59013, 59302, 58180, 59010, 59427, 58989, 59009, 59478, 58161, 58862, 59592, 59183, 58713, 56680, 57627, 58109, 58765, 55480, 57469, 59075, 59587, 59802, 58640, 55292, 57837, 58327, 59442, 59634, 59357, 57563, 59187, 58774, 59148, 58930, 58629, 58981, 56364, 56821, 57042, 56797, 58492, 57547, 56935]
chromosome 1:
[55515, 52712, 50972, 57202, 44434, 58474, 57148, 58601, 59432, 58922, 58811, 56837, 56307, 58979, 57587, 55676, 55843, 56063, 53698, 53256, 57583, 55506, 56703, 56645, 53733, 51647, 58312, 59057, 58276, 56882, 54155, 55442, 39160, 55272, 56413, 59056, 59191, 58727, 56165, 57713, 58063, 56139, 56686, 54421, 58564, 57694, 55352, 54643, 57247, 57190, 56100, 56823, 54598, 54599, 55171, 54748, 51828, 53603, 58670, 59580, 59056, 56262, 55894, 55889, 54023, 55283, 58989, 59537, 59867, 57768, 56005, 55419, 56048, 59513, 58170, 56441, 59464, 55553, 56311, 56500, 56127, 57544, 58986, 59513, 58104, 57665, 58320, 57316, 58143, 55433, 54731, 55889, 56546, 57504, 57357, 59929, 58507, 59870, 58357, 57728, 59129, 56540, 56308, 56679, 56082, 54963, 59070, 57772, 57160, 58211, 59251, 59509, 59232, 59725, 59677, 59752, 58808, 59484, 58370, 59596, 58760, 59313, 59224, 56269, 56414, 58552, 57570, 59776, 58749, 58191, 57618, 58769, 59450, 58836, 57254, 59557, 59111, 58107, 59387, 58429, 57556, 59034, 59377, 59421, 59417, 59066, 59216, 59175, 59481, 58921, 58955, 57438, 59907, 59409, 58760, 56690, 58795, 59490, 58953, 59062, 59056, 59371, 59376, 59009, 59643, 59950, 58731, 59702, 59157, 59006, 59206, 57701, 58972, 58093, 59738, 59753, 59664, 58200, 58665, 58704, 59528, 58763, 57426, 56863, 56730, 56686, 57979, 59164, 58944, 58507, 59261, 59692, 59094, 59265, 58815, 59703, 59092, 59558, 59065, 57698, 58295, 58952, 59088, 59284, 58396, 58648, 57600, 36710, 59132, 57734, 58810, 58591, 58949, 59511, 59862, 59472, 57980, 57753, 56007, 58515, 58940, 59277, 58866, 59355, 58444, 55597, 56852, 57477, 59022, 57547, 60142, 59065, 59028, 59269, 58488, 59849, 59118, 57750, 58680, 57139, 57987, 57744, 58746, 21724, 4395, 1823, 2841, 3345, 4473, 3609, 3939, 4621, 4324, 25306, 7203, 41546, 36016, 44333, 10304, 6660, 5101, 4330, 3781, 3628, 5572, 3902, 4288, 4950, 4445, 4486, 3255, 3556, 3347, 3500, 3818, 3640, 4176, 3630, 3462, 5069, 5335, 5735, 4510, 5545, 33473, 58482, 54795, 55603, 49905, 55722, 51546, 58881, 52438, 58275, 58064, 57014, 58667, 52754, 55320, 53987, 53777, 56562, 55523, 58223, 57521, 56358, 53505, 57114, 55605, 51381, 54740, 57231, 58607, 59056, 57866, 59218, 58191, 58359, 58900, 58780, 55658, 28624, 59037, 58184, 59422, 59407, 59347, 58606, 59605, 59975, 58271, 57703, 58961, 58554, 58121, 58714, 59220, 58610, 58394, 58925, 59040, 57853, 58780, 59224, 58861, 58555, 56106, 56200, 57431, 60018, 59040, 58441, 59997, 59635, 58920, 58680, 57418, 56211, 57933, 58679, 57695, 59569, 59173, 58283, 58235, 58345, 59450, 59036, 58639, 58582, 59429, 58612, 59300, 58949, 59053, 59375, 58926, 58333, 59369, 59495, 58820, 59122, 59173, 59065, 58816, 58978, 58981, 58844, 59119, 59034, 59385, 59148, 57860, 59105, 59184, 59239, 59675, 58961, 57201, 56456, 58281, 58379, 55771, 58471, 57838, 57554, 55699, 58541, 58470, 57597, 58631, 58810, 59020, 56457, 59919, 59813, 59095, 59744, 58688, 58765, 57110, 57766, 57442, 58263, 56998, 59419, 60051, 58834, 59105, 59432, 59561, 59701, 59713, 58766, 59227, 59025, 59567, 59652, 56674, 59089, 59526, 59658, 59613, 58600, 58662, 58453, 55724, 58152, 57318, 57169, 55129, 58672, 58183, 58527, 58088, 30038, 56651, 57674, 59278, 59249, 58883, 57623, 58906, 59754, 58832, 58942, 58974, 58561, 58303, 55014, 56817, 56891, 57110, 58517, 59410, 59003, 59189, 59168, 58708, 58115, 57852, 59062, 57113, 58562, 57965, 58536, 58577, 57959, 56084, 57377, 58678, 57209, 55468, 57069, 56287]
'''
