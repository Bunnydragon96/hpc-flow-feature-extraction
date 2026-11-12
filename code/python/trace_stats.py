"""Flow-size statistics of a batch (how many mice, how big the elephants are).

    python python/trace_stats.py ../data/mawi_60s.pkt > ../results/trace_stats.json
"""
import json
import sys

import numpy as np

import flowlib

out = {}
for path in sys.argv[1:]:
    pk = flowlib.read_pkt(path)
    recs, der, _, _ = flowlib.extract(pk, "omp-part")
    n = np.sort(recs["n"].astype(np.int64))[::-1]
    total = int(n.sum())
    cum = np.cumsum(n)
    out[path.split("/")[-1]] = {
        "packets": total, "flows": int(len(n)),
        "single_packet_flows": int((n == 1).sum()),
        "flows_up_to_3_packets": int((n <= 3).sum()),
        "largest_flow_packets": int(n[0]),
        "flows_for_half_of_packets": int(np.searchsorted(cum, total / 2) + 1),
        "top_1_percent_flows_packet_share": float(cum[max(1, len(n) // 100) - 1] / total),
        "mean_packets_per_flow": total / len(n),
        "bytes": int(recs["bytes"].sum()),
        "protocol_packets": {str(int(p)): int(recs["n"][recs["proto"] == p].sum()) for p in (1, 6, 17, 47)},
    }
json.dump(out, sys.stdout, indent=2)
print()
