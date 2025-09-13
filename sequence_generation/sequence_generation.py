"""
Main DNA sequence generation functions that combine sampling and reconstruction
"""

import numpy as np
from .sampling import (
    generate_kmers, 
    sample_simplex_with_entropy, 
    hit_and_run_sample
)
from .reconstruction import (
    determine_kmer_counts_simple,
    determine_kmer_counts_balanced,
    build_de_bruijn_graph,
    balance_graph,
    find_eulerian_path,
    reconstruct_sequence
)


def generate_dna_sequence_fast(k, L, p_input=None, target_entropy=None):
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
            target_entropy = (0.94 * k) * np.log(4)/np.log(2)
        p = sample_simplex_with_entropy(k, target_entropy=target_entropy)
    else:
        # For slider mode: convert via softmax
        p_input = np.array(p_input)
        p = np.exp(p_input) / np.sum(np.exp(p_input))
    
    # Use balanced counting for better quality
    kmer_counts = determine_kmer_counts_balanced(p, kmers, L, k)
    graph = build_de_bruijn_graph(kmer_counts)
    balanced_graph, _ = balance_graph(graph)
    eulerian_path = find_eulerian_path(balanced_graph)
    sequence = reconstruct_sequence(eulerian_path)
    
    return sequence, np.array(list(kmer_counts.values())), p


def generate_dna_sequence_rigorous(k, L, f_0=1):
    """
    Rigorous DNA sequence generation using hit-and-run sampling.
    Guarantees marginal constraint satisfaction but is slower.
    Suitable for research applications where theoretical correctness is paramount.
    
    Args:
        k: k-mer length
        L: sequence length
        f_0: unused parameter (kept for compatibility)
    
    Returns:
        sequence: generated DNA sequence
        e_n_art: statistic related to artificial edges
        p: k-mer probability distribution used
        kmer_to_index: mapping from k-mers to indices
        all_kmers: list of all k-mers
    """
    kmers = generate_kmers(k)
    p, kmer_to_index, all_kmers = hit_and_run_sample(k, iterations=10000)
    kmer_counts = determine_kmer_counts_balanced(p, kmers, L, k)
    graph = build_de_bruijn_graph(kmer_counts)
    balanced_graph, n_art = balance_graph(graph)
    eulerian_path = find_eulerian_path(balanced_graph)
    sequence = reconstruct_sequence(eulerian_path)
    
    total_edges = sum(len(neighbors) for neighbors in balanced_graph.values())
    e_n_art = 2 * n_art / (total_edges + n_art) if (total_edges + n_art) > 0 else 0
    
    return sequence, e_n_art, p, kmer_to_index, all_kmers


# Convenience alias for the web app
generate_dna_sequence = generate_dna_sequence_fast
