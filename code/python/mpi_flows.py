"""Distributed feature extraction of PCAP batches with MPI4Py.

Every batch (.pkt file made by pcap2pkt or `flowfeat --save`) is an independent
unit of work: flows are defined inside a batch. Each MPI process reads its
batches, extracts the flow features with the C++ library (sequential, OpenMP
or CUDA kernel) and reports to process 0.

    mpirun -n 4 python mpi_flows.py data/batches/*.pkt
    mpirun -n 4 python mpi_flows.py data/batches/*.pkt --schedule dynamic --collect features

Schedules
    static   process r takes a contiguous block of the batch list
    dynamic  every process takes the next unclaimed batch from a shared counter
             (MPI one-sided fetch-and-add), largest batches first
Collect
    summary  only packet/flow counts and a checksum per batch go to process 0
    features all flow records (176 bytes) and derived features (72 bytes) are
             sent to process 0 with Gatherv
"""
import argparse
import json
import os
import sys

import numpy as np
from mpi4py import MPI

import flowlib


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    ap = argparse.ArgumentParser()
    ap.add_argument("batches", nargs="+")
    ap.add_argument("--impl", default="seq", choices=list(flowlib.IMPLS))
    ap.add_argument("--threads", type=int, default=1, help="OpenMP threads per process")
    ap.add_argument("--schedule", default="static", choices=["static", "dynamic"])
    ap.add_argument("--collect", default="summary", choices=["summary", "features"])
    ap.add_argument("--gap", type=int, default=flowlib.DEFAULT_GAP_US)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    files = sorted(args.batches)
    nb = len(files)
    flowlib.lib()  # load the library before timing

    # Shared counter for the dynamic schedule (lives in process 0's window).
    win = None
    if args.schedule == "dynamic":
        win = MPI.Win.Allocate(8 if rank == 0 else 0, 8, comm=comm)
        if rank == 0:
            np.frombuffer(win.tomemory(), dtype="i8")[0] = 0
        order = None
        if rank == 0:  # largest batches first
            order = sorted(range(nb), key=lambda i: -os.path.getsize(files[i]))
        order = comm.bcast(order, root=0)

    comm.Barrier()
    t_start = MPI.Wtime()

    done, recs_parts, der_parts = [], [], []
    t_read = t_compute = t_sched = 0.0

    def work(i):
        nonlocal t_read, t_compute
        if args.collect == "features":
            t0 = MPI.Wtime()
            pk = flowlib.read_pkt(files[i])
            t1 = MPI.Wtime()
            recs, der, _, chk = flowlib.extract(pk, args.impl, args.threads, args.gap, sort=False)
            t2 = MPI.Wtime()
            recs_parts.append(recs)
            der_parts.append(der)
            done.append((i, len(pk), len(recs), chk))
            t_read += t1 - t0
            t_compute += t2 - t1
        else:
            r = flowlib.process_file(files[i], args.impl, args.threads, args.gap)
            done.append((i, r["packets"], r["flows"], r["checksum"]))
            t_read += r["read_s"]
            t_compute += r["compute_s"]

    if args.schedule == "static":
        lo, hi = nb * rank // size, nb * (rank + 1) // size
        for i in range(lo, hi):
            work(i)
    else:
        one = np.array([1], dtype="i8")
        got = np.zeros(1, dtype="i8")
        while True:
            t0 = MPI.Wtime()
            win.Lock(0, MPI.LOCK_SHARED)
            win.Fetch_and_op(one, got, 0, 0, MPI.SUM)
            win.Unlock(0)
            t_sched += MPI.Wtime() - t0
            if got[0] >= nb:
                break
            work(order[int(got[0])])
    t_busy = MPI.Wtime() - t_start

    # ---- collect at process 0
    t0 = MPI.Wtime()
    all_done = comm.gather(done, root=0)
    stats = comm.gather((t_read, t_compute, t_sched, t_busy, len(done)), root=0)
    total_flows_collected = 0
    if args.collect == "features":
        recs = np.concatenate(recs_parts) if recs_parts else np.empty(0, dtype=flowlib.REC)
        der = np.concatenate(der_parts) if der_parts else np.empty(0, dtype=flowlib.DERIVED)
        counts = comm.gather(len(recs), root=0)
        t_rec = MPI.BYTE.Create_contiguous(flowlib.REC.itemsize).Commit()
        t_der = MPI.BYTE.Create_contiguous(flowlib.DERIVED.itemsize).Commit()
        if rank == 0:
            total = sum(counts)
            displs = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
            all_recs = np.empty(total, dtype=flowlib.REC)
            all_der = np.empty(total, dtype=flowlib.DERIVED)
            comm.Gatherv([recs, len(recs), t_rec], [all_recs, counts, displs, t_rec], root=0)
            comm.Gatherv([der, len(der), t_der], [all_der, counts, displs, t_der], root=0)
            total_flows_collected = total
        else:
            comm.Gatherv([recs, len(recs), t_rec], None, root=0)
            comm.Gatherv([der, len(der), t_der], None, root=0)
        t_rec.Free()
        t_der.Free()
    comm.Barrier()
    t_end = MPI.Wtime()
    if win is not None:
        win.Free()

    if rank != 0:
        return
    rows = sorted(x for part in all_done for x in part)
    assert [r[0] for r in rows] == list(range(nb)), "every batch must be processed exactly once"
    packets = sum(r[1] for r in rows)
    flows = sum(r[2] for r in rows)
    check = 0
    for r in rows:
        check = (check + r[3]) & 0xFFFFFFFFFFFFFFFF
    if args.collect == "features":
        assert total_flows_collected == flows
    wall = t_end - t_start
    busy = [s[3] for s in stats]
    result = {
        "bench": "mpi", "impl": args.impl, "procs": size, "threads": args.threads, "schedule": args.schedule,
        "collect": args.collect, "batches": nb, "packets": packets, "flows": flows,
        "checksum": "%016x" % check, "wall_s": wall, "collect_s": t_end - t0,
        "busy_max_s": max(busy), "busy_min_s": min(busy), "busy_mean_s": sum(busy) / size,
        "read_sum_s": sum(s[0] for s in stats), "compute_sum_s": sum(s[1] for s in stats),
        "sched_sum_s": sum(s[2] for s in stats), "batches_per_rank": [s[4] for s in stats],
    }
    if args.json:
        print("RESULT " + json.dumps(result))
    else:
        if not args.quiet:
            print("batch  packets     flows      checksum          file")
            for i, npk, nf, chk in rows:
                print("%5d  %-10d  %-9d  %016x  %s" % (i, npk, nf, chk, os.path.basename(files[i])))
        print("%d batches, %d packets, %d flow records, combined checksum %016x" % (nb, packets, flows, check))
        print("%d processes x %d threads (%s), %s schedule, collect %s" %
              (size, args.threads, args.impl, args.schedule, args.collect))
        print("wall time %.4f s (%.2f million packets/s); collect %.4f s; busy time per process min %.4f / max %.4f s" %
              (wall, packets / wall / 1e6, t_end - t0, min(busy), max(busy)))
        print("batches per process:", result["batches_per_rank"])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
