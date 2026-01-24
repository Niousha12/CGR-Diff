import collections

import math
import numpy as np


class CGR:
    def __init__(self, seq, k_mer):
        self.data = seq
        self.k_mer = k_mer

    def count_kmers(self):
        kmer_dict = collections.defaultdict(int)
        for i in range(len(self.data) - (self.k_mer - 1)):
            kmer_dict[self.data[i:i + self.k_mer]] += 1

        for kmer_key in list(kmer_dict.keys()):
            if "N" in kmer_key:
                del kmer_dict[kmer_key]
        return kmer_dict

    def probabilities(self):  # k-mers count / length of the whole dna
        probabilities = collections.defaultdict(float)
        if self.k_mer == 0:
            probabilities[self.data] = 1.0
        else:
            kmer_dict = self.count_kmers()
            kmer_dict_sum = sum(kmer_dict.values())
            for key, value in kmer_dict.items():
                probabilities[key] = float(value) / kmer_dict_sum  # / (len(self.data) - self.k_mer + 1)
        return probabilities

    def get_fcgr(self):
        probabilities = self.count_kmers()
        array_size = int(math.sqrt(4 ** self.k_mer))
        chaos = []
        for _ in range(array_size):
            chaos.append([0] * array_size)
        max_x = array_size
        max_y = array_size
        pos_x = 1
        pos_y = 1
        for key, value in probabilities.items():
            for char in reversed(key):
                if char == "G" or char == "g":  # T
                    pos_x += max_x / 2
                elif char == "A" or char == "a":  # C
                    pos_y += max_y / 2
                elif char == "T" or char == "t":  # G
                    pos_x += max_x / 2
                    pos_y += max_y / 2
                max_x /= 2
                max_y /= 2

            chaos[int(pos_y - 1)][int(pos_x - 1)] = value
            max_x = array_size
            max_y = array_size
            pos_x = 1
            pos_y = 1

        c = np.array(chaos)
        return c

    def get_fcgr_fast(self, dtype=np.uint32):
        k = self.k_mer
        n = len(self.data)
        size = 1 << k  # 2**k (since sqrt(4**k) = 2**k)

        fcgr = np.zeros((size, size), dtype=dtype)

        # Your quadrant logic implies:
        # x-bit = 1 for {G,T}, y-bit = 1 for {A,T}
        bx_map = {'C': 0, 'A': 0, 'G': 1, 'T': 1}
        by_map = {'C': 0, 'G': 0, 'A': 1, 'T': 1}

        maskbit = 1 << (k - 1)
        x = y = 0
        valid = 0  # how many consecutive A/C/G/T since last N

        for ch in self.data.upper():
            if ch in bx_map:
                bx = bx_map[ch]
                by = by_map[ch]

                # Right shift because your code uses reversed(kmer):
                # newest char becomes the MSB each step
                x = (x >> 1) | (bx * maskbit)
                y = (y >> 1) | (by * maskbit)
                valid += 1

                if valid >= k:
                    fcgr[y, x] += 1
            else:
                # N (or anything else) breaks the window
                x = y = 0
                valid = 0

        return fcgr

    def get_kmer_matrix(self):
        array_size = int(math.sqrt(4 ** self.k_mer))
        kmer_matrix = [[""] * array_size for _ in range(array_size)]

        def recursive_fill(prefix, x, y, size, k):
            if k == 0:
                kmer_matrix[y][x] = prefix
                return
            half = size // 2
            # A (top-left)
            recursive_fill(prefix + "C", x, y, half, k - 1)
            # C (top-right)
            recursive_fill(prefix + "G", x + half, y, half, k - 1)
            # G (bottom-left)
            recursive_fill(prefix + "A", x, y + half, half, k - 1)
            # T (bottom-right)
            recursive_fill(prefix + "T", x + half, y + half, half, k - 1)

        recursive_fill("", 0, 0, array_size, self.k_mer)
        return np.array(kmer_matrix)

    def get_cgr(self):
        seq = list(self.data)
        d = {'C': (0, 0), 'A': (0, 1), 'T': (1, 1), 'G': (1, 0)}
        size = 2 ** self.k_mer
        mat = np.zeros((2 ** self.k_mer, 2 ** self.k_mer))
        x, y = 0.5, 0.5

        for nucleotide in seq:
            if nucleotide in d:
                x += 0.5 * (d[nucleotide][0] - x)
                y += 0.5 * (d[nucleotide][1] - y)
                mat[int(y * size) - 1][int(x * size) - 1] = 1
        return mat

    def get_scatter_cgr(self):
        seq = list(self.data)
        d = {'A': (0, 0), 'C': (0, 1), 'G': (1, 1), 'T': (1, 0)}
        x, y = 0.5, 0.5
        X, Y = [], []
        i = 0

        while seq:
            nucleotide = seq.pop(0)
            if nucleotide in d:
                x += 0.5 * (d[nucleotide][0] - x)
                y += 0.5 * (d[nucleotide][1] - y)
                X.append(x)
                Y.append(y)
                i += 1
        return X, Y

    @staticmethod
    def normalize(input_matrix, clamp_outliers=False, reverse=False):
        # take a copy of the input_matrix to not change the original
        input_matrix_copy = np.copy(input_matrix)
        if clamp_outliers:
            low, high = np.percentile(input_matrix_copy, [2, 98])
            # clamp values to avoid extreme outliers
            input_matrix_copy[input_matrix_copy < low] = low
            input_matrix_copy[input_matrix_copy > high] = high
            # matrix = (input_matrix_copy - low) / (high - low)  # Normalize to 0-1
        if np.max(input_matrix_copy) == 0:
            return input_matrix_copy
        # import matplotlib.pyplot as plt
        # plt.imshow(input_matrix_copy)
        # plt.show()
        matrix = input_matrix_copy - np.min(input_matrix_copy)
        matrix = matrix / np.max(matrix)
        if reverse:
            matrix = 1.0 - matrix  # Invert colors for visualization
        return matrix

    @staticmethod
    def array2img(array, bits=8, resolution=4, m=None, M=None, return_array=False):
        if m is None and M is None:
            m, M = array.min(), array.max()
        if m == M:
            img_rescaled = array
        else:
            # rescale to [0,1]
            img_rescaled = (array - m) / (M - m)

        max_color = resolution ** bits - 1

        # invert colors black->white
        img_array = np.ceil(max_color - img_rescaled * max_color)
        if bits == 8 or bits == 16:
            dtype = eval(f"np.int{bits}")
        else:
            dtype = np.uint8
        img_array = np.array(img_array, dtype=dtype)

        if return_array:
            return img_rescaled, img_array
        return img_array
