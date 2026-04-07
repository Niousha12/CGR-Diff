from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Union

# =========================
# DNA utilities
# =========================
def sanitize(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    if any(c not in "ACGT" for c in seq):
        raise ValueError("Only A/C/G/T supported")
    return seq


# =========================
# Token definitions
# =========================
@dataclass
class Lit:
    s: str


@dataclass
class Copy:
    target: int
    length: int
    source: int


Token = Union[Lit, Copy]


# =========================
# Fast compressor
# =========================
class FastDNACompressor:
    def __init__(
        self,
        min_match: int = 20,
        seed_len: int = 12,
        window: int = 50000,
        max_candidates: int = 32,
    ):
        self.min_match = min_match
        self.seed_len = seed_len
        self.window = window
        self.max_candidates = max_candidates

        if self.seed_len > self.min_match:
            raise ValueError("seed_len should be <= min_match")

    @staticmethod
    def _extend_match(seq: str, i: int, j: int, n: int) -> int:
        """Return exact match length between seq[i:] and seq[j:]."""
        L = 0
        while i + L < n and j + L < i and seq[i + L] == seq[j + L]:
            L += 1
        return L

    def compress(self, seq: str) -> List[Token]:
        seq = sanitize(seq)
        n = len(seq)
        if n == 0:
            return []

        tokens: List[Token] = []
        lit_buf = []

        # k-mer seed -> recent source positions
        index = defaultdict(deque)

        def flush_lit():
            nonlocal lit_buf
            if lit_buf:
                tokens.append(Lit("".join(lit_buf)))
                lit_buf = []

        def add_position(pos: int):
            """Add one position to seed index."""
            if pos + self.seed_len <= n:
                kmer = seq[pos:pos + self.seed_len]
                dq = index[kmer]
                dq.append(pos)

                # Drop entries outside sliding window
                cutoff = pos - self.window
                while dq and dq[0] < cutoff:
                    dq.popleft()

                # Limit number of candidates per k-mer
                while len(dq) > self.max_candidates:
                    dq.popleft()

        i = 0
        while i < n:
            best_source = -1
            best_len = 0

            # Can we try a seed here?
            if i + self.seed_len <= n:
                kmer = seq[i:i + self.seed_len]
                candidates = index.get(kmer, ())

                # Search recent candidates from newest to oldest
                for j in reversed(candidates):
                    if i - j > self.window:
                        continue

                    L = self._extend_match(seq, i, j, n)
                    if L > best_len:
                        best_len = L
                        best_source = j

                if best_len >= self.min_match:
                    flush_lit()
                    tokens.append(Copy(target=i, length=best_len, source=best_source))

                    # Add traversed positions into the index
                    end = min(i + best_len, n)
                    for p in range(i, end):
                        add_position(p)

                    i += best_len
                    continue

            # No good match -> literal
            lit_buf.append(seq[i])
            add_position(i)
            i += 1

        flush_lit()
        return tokens


# =========================
# Decompressor
# =========================
def decompress(tokens: List[Token]) -> str:
    out = []

    for t in tokens:
        if isinstance(t, Lit):
            out.append(t.s)
        elif isinstance(t, Copy):
            built = "".join(out)
            out.append(built[t.source:t.source + t.length])
        else:
            raise TypeError(f"Unknown token type: {type(t)}")

    return "".join(out)


# =========================
# Simple compressed size
# =========================
def compressed_size(tokens: List[Token]) -> int:
    size = 0
    for t in tokens:
        if isinstance(t, Lit):
            size += len(t.s)
        elif isinstance(t, Copy):
            size += 1
    return size


if __name__ == "__main__":
    seq = (
        "ACGTTGCAACGTTGCAACGTTGCA"
        "TTTTTTTT"
        "ACGTTGCAACGTTGCAACGTTGCA"
        "GGGGGGGG"
        "ACGTTGCAACGTTGCAACGTTGCA"
    )

    comp = FastDNACompressor(min_match=8, seed_len=8, window=50000)
    toks = comp.compress(seq)

    print(toks)
    print("Original length:", len(seq))
    print("Compressed size:", compressed_size(toks))

    recon = decompress(toks)
    print("Lossless:", recon == seq)