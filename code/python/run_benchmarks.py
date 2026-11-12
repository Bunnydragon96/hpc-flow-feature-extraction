"""Runs the experiments described in a configuration file and appends one JSON
line per timed sample to <out>/raw_results.jsonl.

    python python/run_benchmarks.py --config configs/media_vm.json --out ../results/media_vm
    python python/run_benchmarks.py --config configs/media_vm.json --out ../results/media_vm --parts size,skew

A part is a list of runs. Each run names a tool and gives its options either as
a "grid" (all combinations) or a "zip" (lists walked in parallel); "fixed"
options are added to every combination. See configs/*.json.
All implementations that process the same input must return the same
checksum; the driver stops with an error if they do not.
"""
import argparse
import glob
import itertools
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def environment():
    env = {"hostname": platform.node(), "system": platform.platform(), "python": platform.python_version(),
           "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    if platform.system() == "Darwin":
        env["cpu"] = sh("sysctl -n machdep.cpu.brand_string")
        env["performance_cores"] = sh("sysctl -n hw.perflevel0.physicalcpu")
        env["efficiency_cores"] = sh("sysctl -n hw.perflevel1.physicalcpu")
        env["memory_bytes"] = sh("sysctl -n hw.memsize")
        env["os"] = sh("sw_vers -productName") + " " + sh("sw_vers -productVersion")
        env["compiler"] = sh("g++-16 --version | head -1")
    else:
        env["cpu"] = sh("lscpu | grep 'Model name' | sed 's/.*: *//'")
        env["logical_cpus"] = sh("nproc")
        env["lscpu"] = sh("lscpu | grep -E 'Thread|Core|Socket|L1d|L2|L3|Hypervisor|Virtualization'")
        env["memory"] = sh("free -h | awk '/Mem:/{print $2}'")
        env["os"] = sh(". /etc/os-release; echo $PRETTY_NAME")
        env["kernel"] = sh("uname -r")
        env["compiler"] = sh("g++ --version | head -1")
        env["gpu"] = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
        env["nvcc"] = sh("nvcc --version | grep release")
        env["thp"] = sh("cat /sys/kernel/mm/transparent_hugepage/enabled")
    env["mpi"] = sh("mpirun --version | head -1")
    env["mpi4py"] = sh(sys.executable + " -c 'import mpi4py; print(mpi4py.__version__)'")
    env["numpy"] = sh(sys.executable + " -c 'import numpy; print(numpy.__version__)'")
    env["loadavg_at_start"] = sh("uptime")
    return env


def combos(run):
    fixed = run.get("fixed", {})
    if "zip" in run:
        keys = list(run["zip"])
        rows = [dict(zip(keys, vals)) for vals in zip(*[run["zip"][k] for k in keys])]
    else:
        keys = list(run.get("grid", {}))
        rows = [dict(zip(keys, vals)) for vals in itertools.product(*[run["grid"][k] for k in keys])]
    for r in rows:
        d = dict(fixed)
        d.update(r)
        yield d


def flag_args(opts, skip=()):
    out = []
    for k, v in opts.items():
        if k in skip:
            continue
        if v is True:
            out.append("--" + k)
        elif v is False or v is None:
            continue
        else:
            out += ["--" + k, str(v)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="")
    ap.add_argument("--data", default=os.path.join(CODE, "..", "data"))
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    raw_path = os.path.join(args.out, "raw_results.jsonl")
    want = [p for p in args.parts.split(",") if p] or list(cfg["parts"])
    reps, warmup = cfg.get("reps", 5), cfg.get("warmup", 1)
    mpirun = cfg.get("mpirun", ["mpirun"])
    json.dump(environment(), open(os.path.join(args.out, "environment.json"), "w"), indent=2)

    # Synthetic batch sets for the MPI experiments.
    for name, spec in cfg.get("batch_sets", {}).items():
        d = os.path.join(args.data, name)
        if not os.path.isdir(d):
            os.makedirs(d)
            for i, logn in enumerate(spec["logn"]):
                subprocess.run([os.path.join(CODE, "flowfeat"), "--logn", str(logn), "--zipf", str(spec["zipf"]),
                                "--seed", str(100 + i), "--save", os.path.join(d, "b%03d.pkt" % i)], check=True,
                               capture_output=True)
            print("created batch set", name, flush=True)

    checks = {}   # input identity -> (checksum, who)
    t_all = time.time()
    with open(raw_path, "a") as raw:
        for part in want:
            t_part = time.time()
            for run in cfg["parts"][part]:
                tool = run["tool"]
                for opts in combos(run):
                    opts = dict(opts)
                    r_reps = run.get("reps", reps)
                    if tool == "flowfeat":
                        o = dict(opts)
                        if "input" in o:
                            o["input"] = os.path.join(args.data, o["input"])
                        cmd = [os.path.join(CODE, "flowfeat")] + flag_args(o) + \
                              ["--json", "--reps", str(r_reps), "--warmup", str(run.get("warmup", warmup))]
                        ident = ("ff", opts.get("input"), opts.get("logn"), opts.get("n"), opts.get("flows"),
                                 opts.get("zipf", 1.0), opts.get("gap"))
                        launches = 1
                    elif tool == "contention":
                        cmd = [os.path.join(CODE, "contention_bench")] + flag_args(opts) + ["--json", "--reps", str(r_reps)]
                        ident, launches = None, 1
                    elif tool == "mpi":
                        files = sorted(glob.glob(os.path.join(args.data, opts["batches"])))[:opts.get("count", None)]
                        cmd = mpirun + ["-n", str(opts["procs"]), sys.executable, os.path.join(HERE, "mpi_flows.py")] + \
                              files + flag_args(opts, skip=("batches", "count", "procs")) + ["--json"]
                        ident, launches = ("mpi", opts["batches"], opts.get("count")), r_reps
                    else:
                        raise SystemExit("unknown tool " + tool)
                    t0 = time.time()
                    samples = []
                    for _ in range(launches):
                        p = subprocess.run(cmd, capture_output=True, text=True, cwd=CODE)
                        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                        if p.returncode != 0 or not lines:
                            print("FAILED:", " ".join(cmd[:12]), "...\n", p.stderr[-600:], flush=True)
                            raise SystemExit(1)
                        for l in lines:
                            rec = json.loads(l[7:])
                            rec["part"] = part
                            if tool == "mpi":
                                rec["batch_set"], rec["count"] = opts["batches"], len(files)
                                rec["rep"] = len(samples)
                            samples.append(rec)
                            raw.write(json.dumps(rec) + "\n")
                    raw.flush()
                    if tool == "contention" and not all(s["ok"] for s in samples):
                        raise SystemExit("contention_bench mismatch: " + " ".join(cmd))
                    if ident is not None:
                        ck = samples[0]["checksum"]
                        if any(s["checksum"] != ck for s in samples) or checks.setdefault(ident, (ck, cmd))[0] != ck:
                            print("CHECKSUM MISMATCH:", " ".join(cmd), ck, checks.get(ident), flush=True)
                            raise SystemExit(1)
                    key = "wall_s" if tool == "mpi" else "total_s"
                    ts = sorted(s[key] for s in samples if s.get("variant", "") != "seq" or tool != "contention")
                    med = ts[len(ts) // 2] if ts else 0.0
                    print("[%s] %-11s %s  median %.4f s  (%d samples, %.1f s)" %
                          (part, tool, " ".join("%s=%s" % kv for kv in opts.items()), med, len(samples),
                           time.time() - t0), flush=True)
            print("== part %s done in %.1f s" % (part, time.time() - t_part), flush=True)
    print("all parts done in %.1f s; results in %s" % (time.time() - t_all, raw_path))


if __name__ == "__main__":
    main()
