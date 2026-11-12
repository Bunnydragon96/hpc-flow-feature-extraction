"""Independent reference implementation in plain Python.

It does not share any code or algorithm with the C++ programs: packets are
grouped with a dictionary of lists, and every feature is computed from the
complete list of packets of the flow with Python integers, fractions and the
standard library. It is slow and is only used by verify.py on small inputs.
"""
import math
from collections import Counter, defaultdict
from fractions import Fraction

NFLAGS, NBINS, BIN_WIDTH = 6, 16, 100


def extract(packets, gap):
    """packets: iterable of (ts, src, dst, ports, meta) in time order.
    Returns {key: (record dict, derived dict)}."""
    groups = defaultdict(list)
    for ts, src, dst, ports, meta in packets:
        groups[(src, dst, ports, meta >> 24)].append((ts, meta & 0xFFFF, (meta >> 16) & 0xFF))
    out = {}
    for key, plist in groups.items():
        times = [p[0] for p in plist]
        lens = [p[1] for p in plist]
        iats = [b - a for a, b in zip(times, times[1:])]
        n = len(plist)
        # bursts: cut the flow wherever the gap between two packets is >= gap
        burst_sizes, size = [], 1
        for iat in iats:
            if iat >= gap:
                burst_sizes.append(size)
                size = 1
            else:
                size += 1
        burst_sizes.append(size)
        hist = Counter(min(l // BIN_WIDTH, NBINS - 1) for l in lens)
        rec = {
            "n": n, "bytes": sum(lens), "sum_len2": sum(l * l for l in lens),
            "len_min": min(lens), "len_max": max(lens),
            "first_ts": times[0], "last_ts": times[-1],
            "iat_min": min(iats) if iats else 0, "iat_max": max(iats) if iats else 0,
            "sum_iat2": sum(i * i for i in iats),
            "bursts": len(burst_sizes), "max_burst": max(burst_sizes),
            "flags": [sum((p[2] >> k) & 1 for p in plist) for k in range(NFLAGS)],
            "hist": [hist.get(b, 0) for b in range(NBINS)],
        }
        dur = times[-1] - times[0]

        def pstd(values):  # population standard deviation, exact until the square root
            if not values:
                return 0.0
            k = len(values)
            mean = Fraction(sum(values), k)
            var = sum((v - mean) ** 2 for v in values) / k
            return math.sqrt(var)

        der = {
            "duration_s": dur * 1e-6,
            "len_mean": sum(lens) / n, "len_std": pstd(lens),
            "iat_mean": (sum(iats) / len(iats)) if iats else 0.0, "iat_std": pstd(iats),
            "pkt_rate": n / (dur * 1e-6) if dur else 0.0,
            "byte_rate": sum(lens) / (dur * 1e-6) if dur else 0.0,
            "entropy": -sum((c / n) * math.log2(c / n) for c in hist.values()),
            "mean_burst": n / len(burst_sizes),
        }
        out[key] = (rec, der)
    return out
