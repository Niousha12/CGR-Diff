import itertools
import random
from collections import defaultdict, Counter, deque

import numpy as np


# ===================================
# Sampling
# ===================================
def generate_kmers(k):
    """Generate all possible k-mers."""
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]


def compute_entropy(prob_dist):
    """Compute Shannon entropy of a probability distribution."""
    prob_dist = np.clip(prob_dist, 1e-12, 1)  # Avoid log(0)
    return -np.sum(prob_dist * np.log(prob_dist) / np.log(2))


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
# Reconstruction
# ===================================
def determine_kmer_counts_balanced(p, kmers, L, k, max_iter=1000, tol=1e-6):
    """
    Convert probabilities to integer k-mer counts while enforcing conservation
    at every (k-1)-mer vertex. More accurate but slightly slower.
    """
    return_dict = {'diff_message': "", 'adjust_message': "", 'counts': {}}
    total_kmers = L - k + 1
    p = np.asarray(p, dtype=float)
    p_dict = dict(zip(kmers, p))
    r = {kmer: total_kmers * p_dict[kmer] for kmer in kmers}

    out_groups = defaultdict(list)
    in_groups = defaultdict(list)
    for kmer in kmers:
        out_groups[kmer[:-1]].append(kmer)
        in_groups[kmer[1:]].append(kmer)

    vertices = set(list(out_groups.keys()) + list(in_groups.keys()))
    for _ in range(max_iter):
        max_diff = 0
        for v in vertices:
            out_sum = sum(r[kmer] for kmer in out_groups.get(v, []))
            in_sum = sum(r[kmer] for kmer in in_groups.get(v, []))
            diff = out_sum - in_sum
            max_diff = max(max_diff, abs(diff))
            if abs(diff) > tol:
                if out_groups.get(v, []):
                    for kmer in out_groups[v]:
                        r[kmer] -= diff / (2 * len(out_groups[v]))
                        if r[kmer] < 0:
                            r[kmer] = 0
                if in_groups.get(v, []):
                    for kmer in in_groups[v]:
                        r[kmer] += diff / (2 * len(in_groups[v]))
        if max_diff < tol:
            break
    else:
        return_dict['adjust_message'] = f"Balance not achieved within max iterations of {max_iter}."

    s = sum(r.values())
    if s > 0:
        for kmer in r:
            r[kmer] *= total_kmers / s

    counts = {}
    for v, group in out_groups.items():
        target = int(round(sum(r[kmer] for kmer in group)))
        for kmer in group:
            counts[kmer] = int(np.floor(r[kmer]))
        current = sum(counts[kmer] for kmer in group)
        rem = target - current
        if rem > 0:
            sorted_group = sorted(group, key=lambda kmer: r[kmer] - np.floor(r[kmer]), reverse=True)
            for kmer in sorted_group[:rem]:
                counts[kmer] += 1

    return_dict['counts'] = counts

    p_final = np.array([counts.get(kmer, 0) / total_kmers for kmer in kmers], dtype=float)
    max_abs_diff = np.max(np.abs(p_final - p))

    # choose one threshold
    final_warning_threshold = 0.01  # 1 percentage point per k-mer

    if max_abs_diff > final_warning_threshold:
        return_dict['diff_message'] = ("The requested k-mer distribution could not be matched exactly, "
                                       "so it was adjusted to generate a valid sequence.")

    return return_dict


def build_de_bruijn_graph(kmer_counts):
    """
    Build a de Bruijn graph from the k-mer counts.
    """
    graph = defaultdict(list)
    for kmer, count in kmer_counts.items():
        if count > 0:
            prefix = kmer[:-1]
            suffix = kmer[1:]
            graph[prefix].extend([suffix] * count)
    for node in graph:
        random.shuffle(graph[node])
    return graph


def compute_vertex_imbalances(graph):
    """
    Compute imbalance δ(v) = outdeg(v) - indeg(v) for each vertex.
    """
    imbalance = defaultdict(int)
    for v in graph:
        imbalance[v] += len(graph[v])
        for u in graph[v]:
            imbalance[u] -= 1
    return imbalance


def balance_graph(graph):
    """
    Balance the de Bruijn graph by adding artificial edges.
    """
    imbalance = compute_vertex_imbalances(graph)
    surplus = {v: imbalance[v] for v in imbalance if imbalance[v] > 0}
    deficit = {v: -imbalance[v] for v in imbalance if imbalance[v] < 0}
    for v in set(list(surplus.keys()) + list(deficit.keys())):
        if v not in graph:
            graph[v] = []
    n_art = 0
    while deficit and surplus:
        u = next(iter(deficit))
        v = next(iter(surplus))
        d = min(deficit[u], surplus[v])
        graph[u].extend([v] * d)
        n_art += d
        deficit[u] -= d
        surplus[v] -= d
        if deficit[u] == 0:
            del deficit[u]
        if surplus[v] == 0:
            del surplus[v]
    return graph, n_art


def find_eulerian_path(graph):
    """Find an Eulerian path in the de Bruijn graph."""
    if not graph:
        return []

    # Find a starting node (preferably one with out-degree > in-degree)
    in_deg = Counter()
    out_deg = Counter()
    for node in graph:
        out_deg[node] = len(graph[node])
        for neighbor in graph[node]:
            in_deg[neighbor] += 1

    start_node = None
    for node in set(in_deg.keys()).union(out_deg.keys()):
        if out_deg[node] > in_deg[node]:
            start_node = node
            break
    if not start_node:
        start_node = next(iter(graph))

    path = []
    stack = [start_node]
    local_graph = {node: deque(neighbors) for node, neighbors in graph.items()}

    while stack:
        current = stack[-1]
        if local_graph.get(current) and local_graph[current]:
            next_node = local_graph[current].popleft()
            stack.append(next_node)
        else:
            path.append(stack.pop())
    return path[::-1]


def reconstruct_sequence(path):
    """Reconstruct a sequence from an Eulerian path."""
    if not path:
        return ""
    sequence = path[0]
    for node in path[1:]:
        sequence += node[-1]
    return sequence


# ===================================
# Generate DNA Sequence
# ===================================
def generate_dna_sequence(k, L, p_input=None, target_entropy=None):
    """
    Fast DNA sequence generation using entropy-based sampling and simple counting.
    Suitable for web applications where speed is important.
    
    Args:
        k: k-mer length
        L: sequence length
        p_input: optional input distribution (for slider mode)
        target_entropy: optional target entropy for sampling
    
    Returns:
        sequence: generated DNA sequence
        kmer_counts: array of k-mer counts
        p: k-mer probability distribution used
    """
    kmers = generate_kmers(k)

    if p_input is None:
        # Use entropy-based sampling
        if target_entropy is None:
            target_entropy = (0.94 * k) * np.log(4) / np.log(2)
        p = sample_simplex_with_entropy(k, target_entropy=target_entropy)
    else:
        # For slider mode: convert via softmax
        p_input = np.array(p_input)
        p = np.exp(p_input) / np.sum(np.exp(p_input))

    # Use balanced counting for better quality
    kmer_counts = determine_kmer_counts_balanced(p, kmers, L, k)
    graph = build_de_bruijn_graph(kmer_counts['counts'])
    balanced_graph, _ = balance_graph(graph)
    eulerian_path = find_eulerian_path(balanced_graph)
    sequence = reconstruct_sequence(eulerian_path)

    return sequence, kmer_counts, p
