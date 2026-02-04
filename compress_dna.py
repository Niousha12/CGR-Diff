import random
from dataclasses import dataclass
from typing import List, Tuple, Union

# =========================
# DNA utilities
# =========================
_COMP = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A"})


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def sanitize(seq: str) -> str:
    seq = "".join(seq.split()).upper()
    if any(c not in "ACGT" for c in seq):
        raise ValueError("Only A/C/G/T supported")
    return seq


def hamming_match(a: str, b: str, max_mm: int):
    edits = []
    mm = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            mm += 1
            if mm > max_mm:
                return None
            edits.append((i, x))  # target base
    return edits


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


@dataclass
class ApproxCopy:
    target: int
    length: int
    source: int
    edits: List[Tuple[int, str]]  # (offset, new_base)


@dataclass
class RCCopy:
    target: int
    length: int
    source: int


@dataclass
class RCApproxCopy:
    target: int
    length: int
    source: int
    edits: List[Tuple[int, str]]


Token = Union[Lit, Copy, ApproxCopy, RCCopy, RCApproxCopy]


# =========================
# Compressor
# =========================
class SimpleDNACompressor:
    def __init__(
            self,
            min_match=20,
            max_mismatches=2,
            window=5000,
            allow_rc=True,
    ):
        self.min_match = min_match
        self.max_mismatches = max_mismatches
        self.window = window
        self.allow_rc = allow_rc

    def compress(self, seq: str) -> List[Token]:
        seq = sanitize(seq)
        out = ""
        tokens: List[Token] = []
        lit_buf = []

        def flush_lit():
            nonlocal lit_buf
            if lit_buf:
                tokens.append(Lit("".join(lit_buf)))
                lit_buf = []

        i = 0
        n = len(seq)

        while i < n:
            best = None  # (token, length)

            start = max(0, len(out) - self.window)

            for j in range(start, len(out)):
                max_len = min(n - i, len(out) - j)

                for L in range(max_len, self.min_match - 1, -1):
                    src = out[j:j + L]
                    tgt = seq[i:i + L]

                    # Exact
                    if src == tgt:
                        best = Copy(i, L, j)
                        break

                    # Approx
                    edits = hamming_match(tgt, src, self.max_mismatches)
                    if edits is not None:
                        best = ApproxCopy(i, L, j, edits)
                        break

                    # Reverse complement
                    if self.allow_rc:
                        src_rc = revcomp(src)
                        if src_rc == tgt:
                            best = RCCopy(i, L, j)
                            break
                        edits = hamming_match(tgt, src_rc, self.max_mismatches)
                        if edits is not None:
                            best = RCApproxCopy(i, L, j, edits)
                            break

                if best:
                    break

            if best:
                flush_lit()
                tok, L = best, best.length
                tokens.append(tok)
                out += seq[i:i + L]
                i += L
            else:
                lit_buf.append(seq[i])
                out += seq[i]
                i += 1

        flush_lit()
        return tokens


# =========================
# Decompressor
# =========================
def decompress(tokens: List[Token]) -> str:
    out = ""

    for t in tokens:
        if isinstance(t, Lit):
            out += t.s

        elif isinstance(t, Copy):
            out += out[-t.dist:-t.dist + t.length]

        elif isinstance(t, ApproxCopy):
            chunk = list(out[-t.dist:-t.dist + t.length])
            for pos, base in t.edits:
                chunk[pos] = base
            out += "".join(chunk)

        elif isinstance(t, RCCopy):
            out += revcomp(out[-t.dist:-t.dist + t.length])

        elif isinstance(t, RCApproxCopy):
            chunk = list(revcomp(out[-t.dist:-t.dist + t.length]))
            for pos, base in t.edits:
                chunk[pos] = base
            out += "".join(chunk)

    return out


# =========================
# Compressed size (C(x))
# =========================
def compressed_size(tokens: List[Token]) -> int:
    """
    Simple deterministic byte cost:
      - Literal: 1 byte per base
      - Copy: 8 bytes
      - ApproxCopy: 8 + 2*len(edits)
    """
    size = 0
    for t in tokens:
        if isinstance(t, Lit):
            size += len(t.s)
        elif isinstance(t, Copy) or isinstance(t, RCCopy):
            size += 1
        elif isinstance(t, ApproxCopy) or isinstance(t, RCApproxCopy):
            size += 1 + len(t.edits)
    return size


if __name__ == "__main__":
    seq = ("ACGTTGCAACGTTGCAACGTTGCA"
           "TTTTTTTT"
           "ACGTTGCAACGTTGCAACGTTGGA"
           + revcomp("ACGTTGCAACGTTGCAACGTTGCA")
           )
    # seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=80))

    comp = SimpleDNACompressor(min_match=8, max_mismatches=2)

    toks = comp.compress(seq)
    size = compressed_size(toks)
    print(toks)
    # recon = decompress(toks)

    print("Original length:", len(seq))
    print("Compressed size (bytes):", compressed_size(toks))
    # print("Lossless:", recon == seq)
