"""Correctness tests. Run from the code folder after `make`:

    python python/verify.py [--pkt FILE ...] [--mpirun mpirun] | tee ../results/verify_output.txt

1. The C++ sequential baseline is compared with an independent plain-Python
   reference (ref_python.py) on small inputs.
2. Every parallel implementation is compared with the sequential baseline:
   integer accumulators must be identical, derived features must agree to a
   relative tolerance of 1e-9. Inputs include edge cases (1 packet, one flow,
   all flows distinct, more threads than packets, odd thread counts, extreme
   burst gaps) and, if given, real traces.
3. GPU launch configurations, the contention micro-benchmark and the MPI4Py
   driver are checked through their command lines.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

import flowlib
import ref_python

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
TOL = 1e-9
passed = failed = 0


def report(ok, name, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [" + detail + "]") if detail else ""))
    sys.stdout.flush()


def max_rel(a, b):
    x = a.view("<f8").reshape(-1, len(flowlib.DERIVED_NAMES))
    y = b.view("<f8").reshape(-1, len(flowlib.DERIVED_NAMES))
    den = np.maximum(np.abs(x), np.abs(y))
    diff = np.abs(x - y)
    rel = np.divide(diff, den, out=np.zeros_like(diff), where=den > 0)
    return float(rel.max()) if rel.size else 0.0


def same(ref, got):
    (r0, d0), (r1, d1) = ref, got
    if len(r0) != len(r1):
        return False, "flow count %d vs %d" % (len(r0), len(r1))
    if r0.tobytes() != r1.tobytes():
        bad = int(np.sum(r0 != r1))
        return False, "%d records differ" % bad
    m = max_rel(d0, d1)
    return m <= TOL, "max relative difference %.1e" % m


def check_python_reference(name, pk, gap):
    recs, der, _, _ = flowlib.extract(pk, "seq", 1, gap)
    ref = ref_python.extract(pk.tolist(), gap)
    ok, detail = len(ref) == len(recs), ""
    worst = 0.0
    if ok:
        for r, d in zip(recs, der):
            key = (int(r["src"]), int(r["dst"]), int(r["ports"]), int(r["proto"]))
            if key not in ref:
                ok, detail = False, "flow missing in the reference"
                break
            rr, dd = ref[key]
            for f in ("n", "bytes", "sum_len2", "len_min", "len_max", "first_ts", "last_ts", "iat_min", "iat_max",
                      "sum_iat2", "bursts", "max_burst"):
                if int(r[f]) != rr[f]:
                    ok, detail = False, "%s: %d vs %d" % (f, int(r[f]), rr[f])
            if list(map(int, r["flags"])) != rr["flags"] or list(map(int, r["hist"])) != rr["hist"]:
                ok, detail = False, "flags or histogram differ"
            for f in flowlib.DERIVED_NAMES:
                a, b = float(d[f]), dd[f]
                rel = abs(a - b) / max(abs(a), abs(b)) if a != b else 0.0
                worst = max(worst, rel)
                if rel > TOL:
                    ok, detail = False, "%s: %.17g vs %.17g" % (f, a, b)
            if not ok:
                break
    else:
        detail = "flow count %d vs %d" % (len(recs), len(ref))
    report(ok, "sequential C++ = Python reference: " + name,
           detail or "%d flows, max relative difference %.1e" % (len(recs), worst))


def cli(args):
    out = subprocess.run(args, capture_output=True, text=True, cwd=CODE)
    res = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    return out.returncode, res, out.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkt", nargs="*", default=[], help="real .pkt batches to include")
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--no-mpi", action="store_true")
    args = ap.parse_args()
    gpu = flowlib.gpu_available()
    print("GPU available: %s" % gpu)

    # ---- 1. independent reference
    for n, flows, zipf, gap in [(1, 1, 1.0, 1000), (2, 1, 1.0, 1000), (3, 2, 0.0, 1000), (5000, 1, 1.0, 1000),
                                (5000, 5000, 0.0, 1000), (20000, 500, 1.0, 1000), (20000, 500, 1.5, 3),
                                (20000, 64, 0.5, 0), (50000, 3000, 1.2, 50), (50000, 200, 2.0, 0xFFFFFFFF)]:
        pk = flowlib.generate(n, flows, zipf, seed=n + flows)
        check_python_reference("n=%d flows=%d zipf=%.1f gap=%d" % (n, flows, zipf, gap), pk, gap)
    for path in args.pkt[:1]:
        pk = flowlib.read_pkt(path)[:60000]
        check_python_reference("first 60000 packets of %s" % os.path.basename(path), pk, 1000)

    # ---- 2. every implementation against the sequential baseline
    cases = []
    for n in (1, 2, 7, 17, 1000, 4097, 100003, 1 << 20):
        for zipf in (0.0, 1.0, 1.8):
            cases.append((n, max(1, n // 16), zipf, 1000))
    cases += [(100000, 1, 1.0, 1000), (100000, 100000, 0.0, 1000), (300000, 3, 0.3, 1000),
              (1 << 20, 1 << 18, 1.2, 0), (1 << 20, 1 << 10, 1.2, 1), (1 << 20, 1 << 14, 1.0, 0xFFFFFFFF),
              (1 << 22, 1 << 20, 0.0, 1000), (1 << 22, 1 << 12, 1.5, 20)]
    inputs = [("n=%d flows=%d zipf=%.1f gap=%d" % c, flowlib.generate(c[0], c[1], c[2], seed=7), c[3]) for c in cases]
    for path in args.pkt:
        inputs.append((os.path.basename(path), flowlib.read_pkt(path), 1000))
    impls = [("omp-merge", t) for t in (1, 2, 3, 7, 8)] + [("omp-part", t) for t in (1, 2, 3, 7, 8)]
    if gpu:
        impls += [("gpu-flow", 0), ("gpu-atomic", 0), ("gpu-chunk", 0)]
    for name, pk, gap in inputs:
        r0, d0, _, c0 = flowlib.extract(pk, "seq", 1, gap)
        for impl, t in impls:
            r1, d1, _, c1 = flowlib.extract(pk, impl, t, gap)
            ok, detail = same((r0, d0), (r1, d1))
            ok = ok and (c0 == c1)
            report(ok, "%s%s = seq: %s" % (impl, " T=%d" % t if t else "", name), "%d flows, %s" % (len(r0), detail))

    # ---- 3. command-line options
    exe = os.path.join(CODE, "flowfeat")
    opts = [["--impl", "omp-part", "--threads", "5", "--parts", "4"],
            ["--impl", "omp-part", "--threads", "8", "--parts", "14"],
            ["--impl", "omp-part", "--threads", "6", "--static"]]
    if gpu:
        for tpb in (32, 100, 1024):
            for v in ("gpu-flow", "gpu-atomic", "gpu-chunk"):
                opts.append(["--impl", v, "--tpb", str(tpb)])
        for ch in (1, 2, 5, 16, 64, 255, 100000):
            opts.append(["--impl", "gpu-chunk", "--chunk", str(ch)])
        opts += [["--impl", "gpu-atomic", "--pageable"], ["--impl", "gpu-flow", "--pinned-out"],
                 ["--impl", "gpu-chunk", "--pinned-out", "--pageable"], ["--impl", "gpu-chunk", "--pageable", "--logn", "23"]]
    for o in opts:
        for zipf in ("0.2", "1.4"):
            base = [exe, "--logn", "21", "--zipf", zipf, "--check", "--json", "--reps", "2"]
            rc, res, err = cli(base + o)
            ok = rc == 0 and len(res) == 2 and all(r["checked"] == 1 and r["mismatch"] == 0 for r in res)
            report(ok, "flowfeat %s (zipf %s) = seq" % (" ".join(o), zipf), err[:100])
    for zipf in ("0", "1", "2"):
        out = subprocess.run([os.path.join(CODE, "contention_bench"), "--logn", "21", "--zipf", zipf, "--threads", "7",
                              "--reps", "2", "--json"], capture_output=True, text=True)
        res = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        report(out.returncode == 0 and len(res) == 8 and all(r["ok"] == 1 for r in res),
               "contention_bench lock/atomic/local = seq (zipf %s, 7 threads)" % zipf)

    # ---- 4. MPI4Py driver: any process count, schedule and collection gives the same batches
    if not args.no_mpi:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="ffverify_")
        files = []
        for i in range(9):
            f = os.path.join(tmp, "b%02d.pkt" % i)
            subprocess.run([exe, "--logn", str(15 + i % 4), "--seed", str(i + 1), "--zipf", "1.1", "--save", f],
                           check=True, capture_output=True)
            files.append(f)
        # expected combined checksum from the library, one batch at a time
        expect = 0
        for f in files:
            expect = (expect + flowlib.process_file(f, "seq", 1)["checksum"]) & 0xFFFFFFFFFFFFFFFF
        for p, sched, coll, impl, thr in [(1, "static", "summary", "seq", 1), (3, "static", "summary", "seq", 1),
                                          (4, "dynamic", "summary", "seq", 1), (2, "static", "features", "seq", 1),
                                          (5, "dynamic", "features", "omp-part", 2), (12, "dynamic", "summary", "omp-merge", 1)]:
            cmd = [args.mpirun, "--oversubscribe", "-n", str(p), sys.executable, os.path.join(HERE, "mpi_flows.py")] + \
                  files + ["--schedule", sched, "--collect", coll, "--impl", impl, "--threads", str(thr), "--json"]
            rc, res, err = cli(cmd)
            ok = rc == 0 and len(res) == 1 and res[0]["checksum"] == "%016x" % expect
            report(ok, "mpi_flows p=%d %s %s %s x%d = one process" % (p, sched, coll, impl, thr), err[-120:] if not ok else "")
        for f in files:
            os.remove(f)
        os.rmdir(tmp)

    print("\n%d checks passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
