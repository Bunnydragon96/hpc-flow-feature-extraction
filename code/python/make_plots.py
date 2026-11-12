"""Turns the raw benchmark results into the report's figures, table bodies,
headline numbers and a summary CSV. Reads only results/*/raw_results.jsonl.

    python python/make_plots.py            (from the code folder)

Outputs: report/figures/*.pdf, report/tables/*.tex, results/summary.csv
"""
import collections
import csv
import json
import os
import statistics

import plot_style as ps
from plot_style import plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
TAB = os.path.join(ROOT, "report", "tables")
COL_W, FULL_W = 3.37, 7.0   # column and page width of the two-column report, inches

# One colour and one marker per implementation, the same in every figure.
IMPLS = ["seq", "omp-merge", "omp-part", "gpu-flow", "gpu-atomic", "gpu-chunk"]
STYLE = {
    "seq":        dict(color="#52514e", marker="s", linestyle="--"),
    "omp-part":   dict(color="#2a78d6", marker="o"),
    "omp-merge":  dict(color="#eb6834", marker="D"),
    "gpu-chunk":  dict(color="#1baf7a", marker="^"),
    "gpu-atomic": dict(color="#eda100", marker="v"),
    "gpu-flow":   dict(color="#e87ba4", marker="X"),
}
LABEL = {"seq": "sequential", "omp-part": "omp-part (8 threads)", "omp-merge": "omp-merge (8 threads)",
         "gpu-chunk": "gpu-chunk", "gpu-atomic": "gpu-atomic", "gpu-flow": "gpu-flow"}
MARK = dict(markeredgecolor="white", markeredgewidth=0.7, markersize=5.5)


def load(host):
    path = os.path.join(RES, host, "raw_results.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def sel(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            rv = r.get(k)
            if isinstance(v, float):
                ok = rv is not None and abs(rv - v) < 1e-9
            elif isinstance(v, (list, tuple, set)):
                ok = rv in v
            else:
                ok = rv == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def med(rows, key="total_s"):
    vals = [r[key] for r in rows]
    if not vals:
        raise KeyError("no samples")
    return statistics.median(vals)


def med_phase(rows, name):
    return statistics.median([r["phases"].get(name, 0.0) for r in rows])


def base(name):
    return os.path.basename(name) if name else ""


def tex(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def fmt_ms(sec):
    ms = sec * 1e3
    return "%.0f" % ms if ms >= 100 else ("%.1f" % ms if ms >= 10 else "%.2f" % ms)


def write(name, lines):
    with open(os.path.join(TAB, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def pow2(n):
    return r"$2^{%d}$" % (n.bit_length() - 1)


NUM = collections.OrderedDict()   # headline numbers -> report/tables/numbers.tex


def num(name, value):
    NUM[name] = value


# ------------------------------------------------------------------ summary
def summary(vm, mac):
    groups = collections.defaultdict(list)
    for host, rows in (("media_vm", vm), ("mac_m4", mac)):
        for r in rows:
            if r.get("bench") == "contention":
                key = (host, r["part"], "contention-" + r["variant"], r["n"], r["flows_param"], r["zipf"], "", r["threads"], "")
                groups[key].append(r["total_s"])
            elif r.get("bench") == "mpi":
                key = (host, r["part"], "mpi-" + r["impl"], r["packets"], r["flows"], "", r["batch_set"] + ":" + str(r["count"]),
                       r["threads"], "p=%d %s %s" % (r["procs"], r["schedule"], r["collect"]))
                groups[key].append(r["wall_s"])
            else:
                extra = ""
                if r["impl"].startswith("gpu"):
                    extra = "tpb=%d chunk=%d pinned=%d pinned_out=%d" % (r["tpb"], r["chunk"], r["pinned"], r.get("pinned_out", 0))
                key = (host, r["part"], r["impl"], r["n"], r["nflows"], r["zipf"] if not r["input"] else "", base(r["input"]),
                       r["threads"] if r["impl"].startswith("omp") else 1, extra)
                groups[key].append(r["total_s"])
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host", "part", "implementation", "packets", "flows", "zipf", "input", "threads", "options",
                    "samples", "median_s", "min_s", "max_s", "q1_s", "q3_s", "million_packets_per_s"])
        for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
            v = sorted(groups[key])
            q = statistics.quantiles(v, n=4) if len(v) >= 4 else [v[0], statistics.median(v), v[-1]]
            m = statistics.median(v)
            w.writerow(list(key) + [len(v), "%.6f" % m, "%.6f" % v[0], "%.6f" % v[-1], "%.6f" % q[0], "%.6f" % q[2],
                                    "%.3f" % (key[3] / m / 1e6)])


def get(rows, part, impl, **kw):
    """Median total time of one implementation in one part."""
    q = dict(part=part, impl=impl)
    q.update(kw)
    if impl.startswith("omp") and "threads" not in q:
        q["threads"] = 8
    return med(sel(rows, **q))


# ------------------------------------------------------------------ size
def part_size(vm):
    logs = [20, 21, 22, 23, 24, 25]
    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    for impl in IMPLS:
        y = [(1 << k) / get(vm, "size", impl, n=1 << k) / 1e6 for k in logs]
        ax.plot(logs, y, label=LABEL[impl], **STYLE[impl], **MARK)
    ax.set_xticks(logs)
    ax.set_xticklabels([r"$2^{%d}$" % k for k in logs])
    ax.set_xlabel("packets in the batch")
    ax.set_ylabel("million packets per second")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.2), ncol=2, columnspacing=1.0, handlelength=2.2)
    save(fig, "throughput_vs_size.pdf")

    lines = []
    for impl in IMPLS:
        t = [get(vm, "size", impl, n=1 << k) for k in logs]
        ref = get(vm, "size", "seq", n=1 << 25)
        sp = ref / t[-1]
        lines.append("%s & %s & %s \\\\" % (tex(impl), " & ".join(fmt_ms(x) for x in t),
                                            "1.00" if impl == "seq" else "%.2f" % sp))
        key = impl.replace("-", "")
        num("size" + {"seq": "Seq", "ompmerge": "Merge", "omppart": "Part", "gpuflow": "Flow", "gpuatomic": "Atomic",
                      "gpuchunk": "Chunk"}[key] + "Sp", "%.1f" % sp)
    write("size.tex", lines)
    n25 = 1 << 25
    num("sizeSeqMpps", "%.1f" % (n25 / get(vm, "size", "seq", n=n25) / 1e6))
    num("sizeChunkMpps", "%.0f" % (n25 / get(vm, "size", "gpu-chunk", n=n25) / 1e6))
    num("sizePartMpps", "%.0f" % (n25 / get(vm, "size", "omp-part", n=n25) / 1e6))
    num("sizeSeqSmallMpps", "%.1f" % ((1 << 20) / get(vm, "size", "seq", n=1 << 20) / 1e6))


# ------------------------------------------------------------------ skew
def part_skew(vm):
    zs = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    n = 1 << 24
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    for impl in IMPLS:
        y = [n / get(vm, "skew", impl, zipf=z) / 1e6 for z in zs]
        ax.plot(zs, y, label=LABEL[impl], **STYLE[impl], **MARK)
    ax.set_yscale("log")
    ax.set_xticks(zs)
    ax.set_xticklabels(["0", "0.5", "0.8", "1", "1.2", "1.5", "2"])
    ax.set_xlabel("Zipf exponent $s$ (skew)")
    ax.set_ylabel("million packets per second")
    ax.set_yticks([5, 10, 20, 50, 100, 200, 500])
    ax.set_yticklabels(["5", "10", "20", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.legend(loc="upper center", bbox_to_anchor=(0.46, -0.19), ncol=2, columnspacing=1.0, handlelength=2.2)
    save(fig, "throughput_vs_skew.pdf")

    lines = []
    tops, flows = [], []
    for z in zs:
        r = sel(vm, part="skew", impl="seq", zipf=z)[0]
        tops.append(100.0 * r["top_flow_packets"] / r["n"])
        flows.append(r["nflows"])
    lines.append("Flows (thousands) & " + " & ".join("%.0f" % (f / 1e3) for f in flows) + " \\\\")
    lines.append("Largest flow (\\% of packets) & " + " & ".join(("%.3f" % t) if t < 0.01 else ("%.2f" % t if t < 1 else "%.1f" % t) for t in tops) + " \\\\")
    lines.append("\\midrule")
    for impl in IMPLS:
        t = [get(vm, "skew", impl, zipf=z) for z in zs]
        best = [min(get(vm, "skew", i, zipf=z) for i in IMPLS) for z in zs]
        cells = [("\\textbf{%s}" % fmt_ms(x)) if x == b else fmt_ms(x) for x, b in zip(t, best)]
        lines.append("%s & %s \\\\" % (tex(impl), " & ".join(cells)))
    write("skew.tex", lines)
    for z, tag in ((0.0, "Zero"), (1.0, "One"), (1.5, "Hi"), (2.0, "Two")):
        seq = get(vm, "skew", "seq", zipf=z)
        num("skewSeq" + tag, fmt_ms(seq))
        num("skewSeqMpps" + tag, "%.1f" % (n / seq / 1e6))
        for impl, nm in (("omp-part", "Part"), ("omp-merge", "Merge"), ("gpu-flow", "Flow"), ("gpu-atomic", "Atomic"), ("gpu-chunk", "Chunk")):
            num("skew" + nm + "Sp" + tag, "%.1f" % (seq / get(vm, "skew", impl, zipf=z)))
    num("topshareHi", "%.0f" % tops[zs.index(1.5)])
    num("topshareOne", "%.1f" % tops[zs.index(1.0)])
    num("topshareTwo", "%.0f" % tops[zs.index(2.0)])
    num("skewSeqRatio", "%.1f" % (get(vm, "skew", "seq", zipf=0.0) / get(vm, "skew", "seq", zipf=2.0)))
    num("skewFlowSlow", "%.1f" % (get(vm, "skew", "gpu-flow", zipf=2.0) / get(vm, "skew", "gpu-chunk", zipf=2.0)))


# ------------------------------------------------------------------ flows
def part_flows(vm):
    fl = [256, 4096, 65536, 262144, 1048576, 4194304]
    lines = []
    for impl in IMPLS:
        t = [get(vm, "flows", impl, flows_param=f) for f in fl]
        best = [min(get(vm, "flows", i, flows_param=f) for i in IMPLS) for f in fl]
        cells = [("\\textbf{%s}" % fmt_ms(x)) if x == b else fmt_ms(x) for x, b in zip(t, best)]
        lines.append("%s & %s \\\\" % (tex(impl), " & ".join(cells)))
    write("flows.tex", lines)
    s = [get(vm, "flows", "seq", flows_param=f) for f in fl]
    num("flowsSeqRatio", "%.0f" % (s[-1] / s[0]))
    num("flowsSeqNsSmall", "%.1f" % (s[0] / (1 << 24) * 1e9))
    num("flowsSeqNsBig", "%.0f" % (s[-1] / (1 << 24) * 1e9))
    num("flowsPartSpBig", "%.1f" % (s[-1] / get(vm, "flows", "omp-part", flows_param=fl[-1])))
    num("flowsMergeSpSmall", "%.1f" % (s[0] / get(vm, "flows", "omp-merge", flows_param=fl[0])))


# ------------------------------------------------------------------ scaling
def part_scaling(vm, mac):
    zs = [0.0, 1.0, 1.5]
    ts = [1, 2, 3, 4, 6, 8]
    fig, axes = ps.subplots(1, 3, figsize=(FULL_W, 2.35))
    ramp = [ps.RAMP5[0], ps.RAMP5[2], ps.RAMP5[4]]
    marks = ["o", "D", "^"]
    for a, impl in enumerate(["omp-part", "omp-merge"]):
        ax = axes[0][a]
        ax.plot(ts, ts, color=ps.MUTED, linestyle=":", linewidth=1.0, label="ideal")
        for i, z in enumerate(zs):
            seq = get(vm, "threads", "seq", zipf=z)
            y = [seq / get(vm, "threads", impl, zipf=z, threads=t) for t in ts]
            ax.plot(ts, y, color=ramp[i], marker=marks[i], label="$s = %g$" % z, **MARK)
        ax.set_title("(%s) %s" % ("ab"[a], impl), loc="left")
        ax.set_xticks(ts)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup over sequential")
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.2)
    # (c) MPI
    ax = axes[0][2]
    procs = [1, 2, 4, 8]
    ax.plot(procs, procs, color=ps.MUTED, linestyle=":", linewidth=1.0, label="ideal")
    series = [("mpi_strong", "static", "real, static", "#2a78d6", "o", "-"),
              ("mpi_strong", "dynamic", "real, dynamic", "#eb6834", "D", "-"),
              ("mpi_uneven", "static", "uneven, static", "#1baf7a", "^", "-"),
              ("mpi_uneven", "dynamic", "uneven, dynamic", "#eda100", "v", "-")]
    for part, sched, lab, col, mk, ls in series:
        rows = sel(vm, part=part, schedule=sched)
        if not rows:
            continue
        t1 = med(sel(vm, part=part, schedule="static", procs=1), "wall_s")
        y = [t1 / med(sel(rows, procs=p), "wall_s") for p in procs]
        ax.plot(procs, y, color=col, marker=mk, linestyle=ls, label=lab, **MARK)
    ax.set_title("(c) MPI4Py, batches", loc="left")
    ax.set_xticks(procs)
    ax.set_xlabel("processes")
    ax.set_ylabel("speedup over one process")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "scaling.pdf")

    # strong-scaling table: speedup over the baseline and efficiency at 4 and 8 threads
    lines = []
    for impl in ["omp-part", "omp-merge"]:
        for z in zs:
            seq = get(vm, "threads", "seq", zipf=z)
            t = {k: get(vm, "threads", impl, zipf=z, threads=k) for k in ts}
            lines.append("%s & %g & %s & %.2f & %.2f & %.2f & %.2f & %.0f\\%% & %.2f \\\\" % (
                tex(impl), z, fmt_ms(seq), seq / t[1], seq / t[2], seq / t[4], seq / t[8], 100 * seq / t[8] / 8, t[1] / t[8]))
            tag = ("Part" if impl == "omp-part" else "Merge") + {0.0: "Zero", 1.0: "One", 1.5: "Hi"}[z]
            num("ss" + tag + "One", "%.2f" % (seq / t[1]))
            num("ss" + tag + "Four", "%.2f" % (seq / t[4]))
            num("ss" + tag + "Eight", "%.2f" % (seq / t[8]))
            num("ss" + tag + "Self", "%.2f" % (t[1] / t[8]))
            num("ss" + tag + "SelfFour", "%.2f" % (t[1] / t[4]))
            num("ss" + tag + "Eff", "%.0f" % (100 * seq / t[8] / 8))
        if impl == "omp-part":
            lines.append("\\midrule")
    write("strong.tex", lines)

    # weak scaling
    lines = []
    ns = [1 << 22, 1 << 23, 1 << 24, 1 << 25]
    base_t = {}
    for i, (t, n) in enumerate(zip([1, 2, 4, 8], ns)):
        seq = get(vm, "weak", "seq", n=n)
        row = "%d & %s & %s" % (t, pow2(n), fmt_ms(seq))
        for impl in ["omp-part", "omp-merge"]:
            x = get(vm, "weak", impl, n=n, threads=t)
            if i == 0:
                base_t[impl] = x
            row += " & %s & %.0f\\%%" % (fmt_ms(x), 100 * base_t[impl] / x)
            if t == 8:
                num("weak" + ("Part" if impl == "omp-part" else "Merge") + "Eff", "%.0f" % (100 * base_t[impl] / x))
        lines.append(row + " \\\\")
    write("weak.tex", lines)

    # second platform (Apple M4)
    if mac:
        mts = [1, 2, 4, 6, 8, 10]
        lines = []
        for impl in ["omp-part", "omp-merge"]:
            for z in zs:
                seq = get(mac, "threads", "seq", zipf=z)
                t = {k: get(mac, "threads", impl, zipf=z, threads=k) for k in mts}
                lines.append("%s & %g & %s & %s \\\\" % (tex(impl), z, fmt_ms(seq), " & ".join("%.2f" % (seq / t[k]) for k in mts)))
                tag = ("Part" if impl == "omp-part" else "Merge") + {0.0: "Zero", 1.0: "One", 1.5: "Hi"}[z]
                num("mac" + tag + "Four", "%.2f" % (seq / t[4]))
                num("mac" + tag + "Ten", "%.2f" % (seq / t[10]))
                num("mac" + tag + "Best", "%.2f" % max(seq / t[k] for k in mts))
            if impl == "omp-part":
                lines.append("\\midrule")
        seq = get(mac, "real", "seq")
        for impl in ["omp-part", "omp-merge"]:
            t = {k: get(mac, "real", impl, threads=k) for k in mts}
            lines.append(("\\midrule\n" if impl == "omp-part" else "") + "%s & real & %s & %s \\\\" % (
                tex(impl), fmt_ms(seq), " & ".join("%.2f" % (seq / t[k]) for k in mts)))
            num("macReal" + ("Part" if impl == "omp-part" else "Merge") + "Best", "%.2f" % max(seq / t[k] for k in mts))
        write("mac.tex", lines)
        num("macRealSeq", fmt_ms(seq))
        num("macRealSeqMpps", "%.1f" % (sel(mac, part="real", impl="seq")[0]["n"] / seq / 1e6))


# ------------------------------------------------------------------ GPU
def part_gpu(vm):
    # time breakdown of gpu-chunk, nanoseconds per packet
    inputs = [("$s=0$", dict(part="skew", zipf=0.0)), ("$s=1$", dict(part="skew", zipf=1.0)),
              ("$s=1.5$", dict(part="skew", zipf=1.5)), ("real, 60 s", dict(part="real", input_base="mawi_60s.pkt"))]
    comps = [("upload", ["h2d"]), ("group (flow id, sort, gather)", ["flow_id", "sort", "gather"]),
             ("reduce and derive", ["reduce", "finalize"]), ("download", ["d2h", "host_copy"])]
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ypos = list(range(len(inputs)))[::-1]
    shares = {}
    for (lab, q), y in zip(inputs, ypos):
        rows = sel(vm, impl="gpu-chunk", **q)
        n = rows[0]["n"]
        left = 0.0
        tot = med(rows) / n * 1e9
        for (cname, phs), col in zip(comps, ps.CAT4):
            v = sum(med_phase(rows, p) for p in phs) / n * 1e9
            ax.barh(y, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0,
                    label=cname if y == ypos[0] else None)
            shares[(lab, cname)] = v / tot
            left += v
        ax.text(left + 0.4, y, "%.1f" % tot, va="center", ha="left", fontsize=7.5, color=ps.INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([i[0] for i in inputs])
    ax.set_xlabel("nanoseconds per packet (gpu-chunk)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.3), ncol=2, columnspacing=1.0, handlelength=1.2)
    save(fig, "gpu_breakdown.pdf")
    num("gpuRealDownShare", "%.0f" % (100 * shares[("real, 60 s", "download")]))
    num("gpuRealXferShare", "%.0f" % (100 * (shares[("real, 60 s", "download")] + shares[("real, 60 s", "upload")])))
    num("gpuOneXferShare", "%.0f" % (100 * (shares[("$s=1$", "download")] + shares[("$s=1$", "upload")])))
    num("gpuOneGroupShare", "%.0f" % (100 * shares[("$s=1$", "group (flow id, sort, gather)")]))

    # threads per block
    lines = []
    tpbs = [32, 64, 128, 256, 512, 1024]
    for impl in ["gpu-flow", "gpu-atomic", "gpu-chunk"]:
        t = [med(sel(vm, part="gpu_tpb", impl=impl, tpb=b)) for b in tpbs]
        lines.append("%s & %s \\\\" % (tex(impl), " & ".join(("\\textbf{%s}" % fmt_ms(x)) if x == min(t) else fmt_ms(x) for x in t)))
        num("tpbSpread" + impl.split("-")[1].capitalize(), "%.0f" % (100 * (max(t) - min(t)) / min(t)))
    write("gpu_tpb.tex", lines)

    # chunk size: reduce-phase time
    chunks = [1, 2, 4, 8, 16, 32, 64, 128, 255]
    lines = []
    for lab, q in [("$s=0$", dict(zipf=0.0, input="")), ("$s=1$", dict(zipf=1.0, input="")), ("$s=1.5$", dict(zipf=1.5, input="")),
                   ("real", dict(input_base="mawi_60s.pkt"))]:
        t = [med_phase(sel(vm, part="gpu_chunk", chunk=c, **q), "reduce") for c in chunks]
        lines.append("%s & %s \\\\" % (lab, " & ".join(("\\textbf{%s}" % fmt_ms(x)) if x == min(t) else fmt_ms(x) for x in t)))
        if lab == "$s=1.5$":
            num("chunkHiOne", fmt_ms(t[0]))
            num("chunkHiBest", fmt_ms(min(t)))
            num("chunkHiLast", fmt_ms(t[-1]))
    write("gpu_chunk.tex", lines)
    # reduce-phase time of the three kernels (skew part)
    lines = []
    for impl in ["gpu-flow", "gpu-atomic", "gpu-chunk"]:
        cells = []
        for z in [0.0, 1.0, 1.5, 2.0]:
            rows = sel(vm, part="skew", impl=impl, zipf=z)
            cells.append(fmt_ms(med_phase(rows, "reduce") + med_phase(rows, "finalize")))
        rows = sel(vm, part="real", impl=impl, input_base="mawi_60s.pkt")
        cells.append(fmt_ms(med_phase(rows, "reduce") + med_phase(rows, "finalize")))
        lines.append("%s & %s \\\\" % (tex(impl), " & ".join(cells)))
    write("gpu_reduce.tex", lines)

    # transfers
    lines = []
    for lab, q in [("$s=1$", dict(input="")), ("real", dict(input_base="mawi_60s.pkt"))]:
        for pin in (1, 0):
            for pout in (0, 1):
                rows = sel(vm, part="transfer", pinned=pin, pinned_out=pout, **q)
                n = rows[0]["n"]
                mb_in = n * 20 / 1e6
                mb_out = rows[0]["nflows"] * 248 / 1e6
                up, down = med_phase(rows, "h2d"), med_phase(rows, "d2h") + med_phase(rows, "host_copy")
                lines.append("%s & %s & %s & %s & %.1f & %s & %.1f & %s \\\\" % (
                    lab, "page-locked" if pin else "pageable", "staged" if pout else "direct",
                    fmt_ms(up), mb_in / up / 1e3, fmt_ms(down), mb_out / down / 1e3, fmt_ms(med(rows))))
                if lab == "real" and pin == 1 and pout == 0:
                    num("xferRealOutMB", "%.0f" % mb_out)
                    num("xferRealInMB", "%.0f" % mb_in)
        if lab == "$s=1$":
            lines.append("\\midrule")
    write("transfer.tex", lines)


# ------------------------------------------------------------------ contention
def part_contention(vm):
    zs = [0.0, 0.5, 1.0, 1.2, 1.5, 2.0]
    n = 1 << 24
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    sty = {"seq": dict(color="#52514e", marker="s", linestyle="--", label="one thread"),
           "lock": dict(color="#e34948", marker="X", label="shared table, locks (8 threads)"),
           "atomic": dict(color="#eda100", marker="v", label="shared table, atomics (8 threads)"),
           "local": dict(color="#2a78d6", marker="o", label="private tables + merge (8 threads)")}
    val = {}
    for v in ["seq", "lock", "atomic", "local"]:
        y = []
        for z in zs:
            rows = sel(vm, part="contention", variant=v, zipf=z, threads=(1 if v == "seq" else 8))
            if v == "seq":   # the reference is run once per launch; pool all launches of this skew
                rows = sel(vm, part="contention", variant=v, zipf=z)
            y.append(n / med(rows) / 1e6)
            val[(v, z)] = y[-1]
        ax.plot(zs, y, **sty[v], **MARK)
    ax.set_yscale("log")
    ax.set_xticks(zs)
    ax.set_xticklabels(["0", "0.5", "1", "1.2", "1.5", "2"])
    ax.set_yticks([10, 20, 50, 100, 200, 500, 1000])
    ax.set_yticklabels(["10", "20", "50", "100", "200", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Zipf exponent $s$ (skew)")
    ax.set_ylabel("million packets per second")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, handlelength=2.2)
    save(fig, "contention.pdf")
    num("contLockZero", "%.1f" % (val[("lock", 0.0)] / val[("seq", 0.0)]))
    num("contAtomicZero", "%.1f" % (val[("atomic", 0.0)] / val[("seq", 0.0)]))
    num("contLockHi", "%.0f" % (val[("seq", 1.5)] / val[("lock", 1.5)]))
    num("contAtomicHi", "%.1f" % (val[("seq", 1.5)] / val[("atomic", 1.5)]))
    num("contLockTwo", "%.0f" % (val[("seq", 2.0)] / val[("lock", 2.0)]))
    num("contLocalHi", "%.1f" % (val[("local", 1.5)] / val[("seq", 1.5)]))
    num("contLocalZero", "%.2f" % (val[("local", 0.0)] / val[("seq", 0.0)]))
    # lock throughput by thread count at s = 1.5
    lines = []
    for v in ["lock", "atomic", "local"]:
        cells = []
        for z in (0.0, 1.5):
            for t in (1, 2, 4, 8):
                cells.append("%.0f" % (n / med(sel(vm, part="contention", variant=v, zipf=z, threads=t)) / 1e6))
        lines.append("%s & %s \\\\" % (v, " & ".join(cells)))
    write("contention.tex", lines)


# ------------------------------------------------------------------ real trace
def part_real(vm):
    files = ["mawi_60s.pkt", "mawi_120s.pkt", "mawi_240s.pkt"]
    info = {}
    for f in files:
        r = sel(vm, part="real", impl="seq", input_base=f)[0]
        info[f] = r
    lines = ["Packets (millions) & " + " & ".join("\\multicolumn{2}{c}{%.1f}" % (info[f]["n"] / 1e6) for f in files) + " \\\\",
             "Flows (millions) & " + " & ".join("\\multicolumn{2}{c}{%.1f}" % (info[f]["nflows"] / 1e6) for f in files) + " \\\\",
             "\\midrule"]
    for impl in IMPLS:
        cells = []
        for f in files:
            rows = sel(vm, part="real", impl=impl, input_base=f)
            if impl.startswith("omp"):
                rows = sel(rows, threads=8)
            if not rows:
                cells.append("-- & --")
                continue
            t = med(rows)
            seq = med(sel(vm, part="real", impl="seq", input_base=f))
            best = min(med(sel(sel(vm, part="real", impl=i, input_base=f), threads=8) if i.startswith("omp") else
                           sel(vm, part="real", impl=i, input_base=f)) for i in IMPLS
                       if sel(vm, part="real", impl=i, input_base=f))
            cells.append("%s & %s" % (fmt_ms(t), ("\\textbf{%.2f}" if t == best else "%.2f") % (seq / t)))
            if f == "mawi_60s.pkt":
                nm = {"seq": "Seq", "omp-merge": "Merge", "omp-part": "Part", "gpu-flow": "Flow", "gpu-atomic": "Atomic", "gpu-chunk": "Chunk"}[impl]
                num("real" + nm + "Sp", "%.2f" % (seq / t))
                num("real" + nm + "Mpps", "%.1f" % (info[f]["n"] / t / 1e6))
        lines.append("%s & %s \\\\" % (tex(impl), " & ".join(cells)))
    write("real.tex", lines)
    r = info["mawi_60s.pkt"]
    num("realN", "%.1f" % (r["n"] / 1e6))
    num("realF", "%.1f" % (r["nflows"] / 1e6))
    num("realTop", "%.1f" % (100.0 * r["top_flow_packets"] / r["n"]))
    num("realGbps", "%.1f" % (r["bytes"] * 8 / 60 / 1e9))
    for impl, nm in (("omp-part", "Part"), ("omp-merge", "Merge")):
        seq = med(sel(vm, part="real", impl="seq", input_base="mawi_60s.pkt"))
        t1 = med(sel(vm, part="real", impl=impl, input_base="mawi_60s.pkt", threads=1))
        t8 = med(sel(vm, part="real", impl=impl, input_base="mawi_60s.pkt", threads=8))
        num("real" + nm + "One", "%.2f" % (seq / t1))
        num("real" + nm + "Self", "%.2f" % (t1 / t8))


# ------------------------------------------------------------------ MPI
def part_mpi(vm):
    procs = [1, 2, 4, 8]
    lines = []

    def wall(part, **kw):
        return med(sel(vm, part=part, **kw), "wall_s")

    t1 = wall("mpi_strong", schedule="static", procs=1)
    u1 = wall("mpi_uneven", schedule="static", procs=1)
    for p in procs:
        a, b = wall("mpi_strong", schedule="static", procs=p), wall("mpi_strong", schedule="dynamic", procs=p)
        c, d = wall("mpi_uneven", schedule="static", procs=p), wall("mpi_uneven", schedule="dynamic", procs=p)
        lines.append("%d & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\" % (p, a, t1 / a, b, t1 / b, c, u1 / c, d, u1 / d))
        if p == 8:
            num("mpiStaticSp", "%.2f" % (t1 / a)); num("mpiDynSp", "%.2f" % (t1 / b))
            num("mpiStaticEff", "%.0f" % (100 * t1 / a / 8)); num("mpiDynEff", "%.0f" % (100 * t1 / b / 8))
            num("mpiUnevenStaticSp", "%.2f" % (u1 / c)); num("mpiUnevenDynSp", "%.2f" % (u1 / d))
        if p == 4:
            num("mpiStaticSpFour", "%.2f" % (t1 / a)); num("mpiDynSpFour", "%.2f" % (t1 / b))
            num("mpiUnevenStaticSpFour", "%.2f" % (u1 / c)); num("mpiUnevenDynSpFour", "%.2f" % (u1 / d))
    write("mpi.tex", lines)
    r = sel(vm, part="mpi_strong", schedule="static", procs=1)[0]
    num("mpiPackets", "%.1f" % (r["packets"] / 1e6))
    num("mpiFlows", "%.1f" % (r["flows"] / 1e6))
    num("mpiOneWall", "%.1f" % t1)
    num("mpiBestMpps", "%.1f" % (r["packets"] / min(wall("mpi_strong", schedule=s, procs=8) for s in ("static", "dynamic")) / 1e6))
    rs = sel(vm, part="mpi_strong", schedule="static", procs=8)
    num("mpiStaticImb", "%.2f" % statistics.median([x["busy_max_s"] / x["busy_mean_s"] for x in rs]))
    ru = sel(vm, part="mpi_uneven", schedule="static", procs=4)
    num("mpiUnevenImb", "%.2f" % statistics.median([x["busy_max_s"] / x["busy_mean_s"] for x in ru]))
    rd = sel(vm, part="mpi_strong", schedule="dynamic", procs=8)
    num("mpiSchedMs", "%.1f" % (1e3 * statistics.median([x["sched_sum_s"] for x in rd])))

    # weak scaling, collection cost, hybrid
    lines = []
    w1 = wall("mpi_weak", procs=1)
    for p, cnt in zip(procs, [3, 6, 12, 24]):
        w = wall("mpi_weak", procs=p)
        s = wall("mpi_collect", collect="summary", procs=p)
        f = wall("mpi_collect", collect="features", procs=p)
        thr = 8 // p
        h = wall("mpi_hybrid", procs=p, threads=thr)
        lines.append("%d & %d & %.2f & %.0f\\%% & %.2f & %.2f & %.0f\\%% & %d\\x%d & %.2f \\\\" % (
            p, cnt, w, 100 * w1 / w, s, f, 100 * (f - s) / f, p, thr, h))
        if p == 8:
            num("mpiWeakEff", "%.0f" % (100 * w1 / w))
            num("mpiCollectShare", "%.0f" % (100 * (f - s) / f))
            num("mpiCollectSlow", "%.1f" % (f / s))
    write("mpi2.tex", lines)
    rf = sel(vm, part="mpi_collect", collect="features", procs=8)[0]
    num("mpiCollectGB", "%.1f" % (rf["flows"] * 248 / 1e9))
    hy = {p: wall("mpi_hybrid", procs=p, threads=8 // p) for p in procs}
    bestp = min(hy, key=hy.get)
    num("mpiHybridBest", "%d\\x%d" % (bestp, 8 // bestp))
    num("mpiHybridBestS", "%.2f" % hy[bestp])
    num("mpiHybridBestMpps", "%.1f" % (r["packets"] / hy[bestp] / 1e6))
    num("mpiHybridSp", "%.1f" % (t1 / hy[bestp]))


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    vm, mac = load("media_vm"), load("mac_m4")
    for r in vm + mac:
        r["input_base"] = base(r.get("input", ""))
    summary(vm, mac)
    done = []
    for name, fn in [("size", lambda: part_size(vm)), ("skew", lambda: part_skew(vm)), ("flows", lambda: part_flows(vm)),
                     ("scaling", lambda: part_scaling(vm, mac)), ("gpu", lambda: part_gpu(vm)),
                     ("contention", lambda: part_contention(vm)), ("real", lambda: part_real(vm)), ("mpi", lambda: part_mpi(vm))]:
        try:
            fn()
            done.append(name)
        except (KeyError, IndexError, statistics.StatisticsError) as e:
            print("skipped %s (data missing: %s)" % (name, e))
    # verification counts
    for host, macro in (("media_vm", "nchecks"), ("mac_m4", "nchecksMac")):
        p = os.path.join(RES, host, "verify_output.txt")
        if os.path.exists(p):
            last = open(p).read().strip().splitlines()[-1]
            num(macro, last.split()[0])
            num(macro + "Failed", last.split()[3])
    p = os.path.join(RES, "trace_stats.json")
    if os.path.exists(p):
        st = json.load(open(p))["mawi_60s.pkt"]
        num("realSingle", "%.0f" % (100.0 * st["single_packet_flows"] / st["flows"]))
        num("realHalfFlows", "{:,}".format(st["flows_for_half_of_packets"]).replace(",", "{,}"))
        num("realHalfPct", "%.2f" % (100.0 * st["flows_for_half_of_packets"] / st["flows"]))
        num("realMeanPk", "%.1f" % st["mean_packets_per_flow"])
    num("nfeat", "25")
    with open(os.path.join(TAB, "numbers.tex"), "w") as f:
        f.write("% Generated by make_plots.py from the raw results. Do not edit.\n")
        for k, v in NUM.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print("done:", ", ".join(done), "| %d numbers" % len(NUM))


if __name__ == "__main__":
    main()
