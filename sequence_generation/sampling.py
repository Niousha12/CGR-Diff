"""
Sampling strategies for k-mer distributions
"""

import numpy as np
import itertools
from scipy.linalg import svd
from collections import defaultdict


def generate_kmers(k):
    """Generate all possible k-mers."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]


def compute_entropy(prob_dist):
    """Compute Shannon entropy of a probability distribution."""
    prob_dist = np.clip(prob_dist, 1e-12, 1)  # Avoid log(0)
    return -np.sum(prob_dist * np.log(prob_dist)/np.log(2))


# ===================================
# ENTROPY-BASED SAMPLING (FAST)
# ===================================

def sample_simplex_with_entropy(k, target_entropy=None, tol=0.01, max_iter=100):
    """
    Sample from the probability simplex using Dirichlet distribution
    with entropy constraints. Fast but doesn't enforce marginal constraints.
    """
    K = 4 ** k
    if target_entropy is None:
        alpha = np.ones(K)
        return np.random.dirichlet(alpha)
    
    alpha_low = 0.01
    alpha_high = 100.0
    for _ in range(max_iter):
        alpha = (alpha_low + alpha_high) / 2
        p = np.random.dirichlet(np.full(K, alpha))
        entropy = compute_entropy(p)
        if abs(entropy - target_entropy) < tol:
            return p
        elif entropy > target_entropy:
            alpha_high = alpha
        else:
            alpha_low = alpha
    return p


# ===================================
# CONSTRAINT-BASED SAMPLING (SLOW BUT CORRECT)
# ===================================

def build_B(k):
    """
    Build the constraint matrix B for a given k:
      - The first row enforces sum(x) = 1.
      - For each (k-1)-mer w (except one dropped to remove redundancy),
        the row enforces: sum_{a in Σ} x(wa) - sum_{a in Σ} x(aw) = 0.
    Returns:
      B: a (4^(k-1)) x (4^k) matrix.
      kmer_to_index: dictionary mapping each k-mer to its index (lexicographic order).
      all_kmers: list of all k-mers in lexicographic order.
    """
    bases = ['A', 'C', 'G', 'T']
    n = 4**k
    kmers_kminus1 = [''.join(p) for p in itertools.product(bases, repeat=k-1)]
    kmers_kminus1.sort()
    kmers_kminus1 = kmers_kminus1[:-1]  # drop one to remove redundancy
    
    B_rows = []
    # Row 0 enforces normalization.
    B_rows.append(np.ones(n))
    
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    all_kmers.sort()
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    for w in kmers_kminus1:
        row = np.zeros(n)
        for a in bases:
            # k-mer with prefix v = w -> coefficient +1
            kmer1 = w + a
            row[kmer_to_index[kmer1]] += 1
            # k-mer with suffix v = w -> coefficient -1
            kmer2 = a + w
            row[kmer_to_index[kmer2]] -= 1
        B_rows.append(row)
        
    B = np.vstack(B_rows)
    return B, kmer_to_index, all_kmers


def compute_nullspace(B):
    """
    Compute an orthonormal basis for the nullspace of B using SVD.
    """
    _, s, Vh = svd(B, full_matrices=True)
    tol_val = max(B.shape) * np.amax(s) * np.finfo(s.dtype).eps
    r = np.sum(s > tol_val)
    nullspace = Vh[r:].T
    return nullspace


def hit_and_run_sample(k, iterations=10000):
    """
    Sample a point in the polytope (subset of the simplex)
    of k-mer frequency vectors that satisfy the conservation constraints.
    More theoretically correct but significantly slower.
    """
    B, kmer_to_index, all_kmers = build_B(k)
    n = 4**k
    N = compute_nullspace(B)
    x = np.ones(n) / n  # start at the uniform distribution
    
    for _ in range(iterations):
        z = np.random.randn(N.shape[1])
        d = N @ z
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            continue
        d = d / norm_d
        
        t_min = -np.inf
        t_max = np.inf
        for i in range(n):
            if abs(d[i]) < 1e-12:
                continue
            t_candidate = -x[i] / d[i]
            if d[i] > 0:
                t_min = max(t_min, t_candidate)
            else:
                t_max = min(t_max, t_candidate)
        if t_min > t_max:
            continue
        t = np.random.uniform(t_min, t_max)
        x = x + t * d
    return x, kmer_to_index, all_kmers
