"""
DNA sequence reconstruction algorithms using de Bruijn graphs
"""

import random
import numpy as np
from collections import defaultdict, deque, Counter


def kmer_index(kmer):
    """
    Convert a k-mer into its lexicographic index.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = 0
    for char in kmer:
        idx = idx * 4 + mapping[char]
    return idx


def compute_empirical_distribution(sequence, k):
    """
    Compute the k-mer frequency distribution from a sequence.
    """
    num_kmers = 4 ** k
    counts = [0] * num_kmers
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        idx = kmer_index(kmer)
        counts[idx] += 1
    total_count = sum(counts)
    if total_count == 0:
        return np.zeros(num_kmers)
    return np.array(counts) / total_count


def reconstruction_error(sequence, k, target_distribution):
    """
    Compute the L1 distance (Total Variation distance) between the empirical 
    k-mer distribution (from sequence) and the target distribution.
    
    Args:
        sequence: The generated DNA sequence
        k: k-mer length
        target_distribution: The target k-mer probability distribution
    
    Returns:
        L1 distance between empirical and target distributions
    """
    emp_dist = compute_empirical_distribution(sequence, k)
    return np.sum(np.abs(target_distribution - emp_dist))


def determine_kmer_counts_simple(p, kmers, L, k):
    """
    Simple rounding-based conversion of probabilities to integer k-mer counts.
    Fast but may not preserve exact marginal constraints.
    """
    total_kmers = L - k + 1
    raw_counts = p * total_kmers
    counts = np.round(raw_counts).astype(int)
    difference = total_kmers - np.sum(counts)
    
    while difference != 0:
        idx = np.random.choice(len(counts))
        if difference > 0:
            counts[idx] += 1
            difference -= 1
        elif counts[idx] > 0:
            counts[idx] -= 1
            difference += 1
    return dict(zip(kmers, counts))


def determine_kmer_counts_balanced(p, kmers, L, k, max_iter=1000, tol=1e-6):
    """
    Convert probabilities to integer k-mer counts while enforcing conservation
    at every (k-1)-mer vertex. More accurate but slightly slower.
    """
    total_kmers = L - k + 1
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
                if in_groups.get(v, []):
                    for kmer in in_groups[v]:
                        r[kmer] += diff / (2 * len(in_groups[v]))
        if max_diff < tol:
            break
    else:
        print("Warning: Balance not achieved within max iterations.")
    
    s = sum(r.values())
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
    return counts


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
