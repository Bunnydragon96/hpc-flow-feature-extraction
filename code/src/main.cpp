// flowfeat: packet-to-flow feature extraction.
//
//   flowfeat                          small demonstration on synthetic traffic
//   flowfeat --impl omp-part --threads 4 --logn 24 --zipf 1.0 --check
//   flowfeat --impl gpu-chunk --input trace.pkt --reps 5 --json
//
// Run with --help for all options.
#include "flowfeat.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <string>

using namespace ff;

static void usage() {
    printf(
        "Usage: flowfeat [options]\n"
        "Input (default: synthetic traffic)\n"
        "  --input FILE     read packets from a .pkt file (see pcap2pkt)\n"
        "  --logn K         synthetic: 2^K packets (default 20)\n"
        "  --n N            synthetic: N packets\n"
        "  --flows F        synthetic: number of flows (default N/16)\n"
        "  --zipf S         synthetic: Zipf exponent of flow popularity, 0 = uniform (default 1.0)\n"
        "  --seed X         synthetic: random seed (default 1)\n"
        "Implementation\n"
        "  --impl NAME      seq | omp-merge | omp-part | gpu-flow | gpu-atomic | gpu-chunk (default seq)\n"
        "  --threads T      OpenMP threads (default: all)\n"
        "  --parts L        omp-part: 2^L partitions (default 10)\n"
        "  --static         omp-part: static schedule instead of dynamic\n"
        "  --tpb B          gpu: threads per block (default 256)\n"
        "  --chunk C        gpu-chunk: packets per thread, 1..255 (default 8)\n"
        "  --pageable       gpu: upload from pageable host memory instead of page-locked\n"
        "  --pinned-out     gpu: download into page-locked memory and copy, instead of directly\n"
        "  --gap US         a gap of at least US microseconds ends a burst (default 1000)\n"
        "Run control\n"
        "  --reps R         timed repetitions (default 1)\n"
        "  --warmup W       untimed repetitions first (default 0)\n"
        "  --check          compare every result with the sequential reference\n"
        "  --json           print one 'RESULT {...}' line per repetition\n"
        "  --quiet          do not print the example flows\n"
        "  --out FILE       write the flow features to a CSV file (sorted by flow key)\n"
        "  --save FILE      write the packets to a .pkt file and exit\n");
}

static Result run_impl(const std::string& impl, const Packet* pk, size_t n, uint32_t gap, int threads,
                       int parts, bool dyn, const GpuOpts& go) {
    if (impl == "seq") return run_seq(pk, n, gap);
    if (impl == "omp-merge") return run_omp_merge(pk, n, gap, threads);
    if (impl == "omp-part") return run_omp_part(pk, n, gap, threads, parts, dyn);
    GpuOpts g = go;
    if (impl == "gpu-flow") g.variant = 0;
    else if (impl == "gpu-atomic") g.variant = 1;
    else if (impl == "gpu-chunk") g.variant = 2;
    else { fprintf(stderr, "unknown --impl %s\n", impl.c_str()); exit(2); }
    return run_gpu(pk, n, gap, g);
}

static std::string ip_str(uint32_t a) {
    char b[32];
    snprintf(b, sizeof b, "%u.%u.%u.%u", a >> 24, (a >> 16) & 255, (a >> 8) & 255, a & 255);
    return b;
}

static bool write_csv(const std::string& path, Result& r) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fprintf(f, "src,dst,sport,dport,proto,packets,bytes,first_us,last_us,duration_s,len_min,len_max,len_mean,"
               "len_std,iat_min_us,iat_max_us,iat_mean_us,iat_std_us,pkt_rate,byte_rate,fin,syn,rst,psh,ack,urg,"
               "len_entropy_bits,bursts,max_burst,mean_burst\n");
    for (size_t j = 0; j < r.recs.size(); j++) {
        const FlowRec& a = r.recs[j];
        const FlowDerived& d = r.derived[j];
        fprintf(f, "%s,%s,%u,%u,%u,%u,%" PRIu64 ",%u,%u,%.6f,%u,%u,%.6f,%.6f,%u,%u,%.6f,%.6f,%.6f,%.6f,%u,%u,%u,%u,%u,%u,%.9f,%u,%u,%.6f\n",
                ip_str(a.src).c_str(), ip_str(a.dst).c_str(), a.ports >> 16, a.ports & 0xFFFF, a.proto, a.n,
                a.bytes, a.first_ts, a.last_ts, d.duration_s, a.len_min, a.len_max, d.len_mean, d.len_std,
                a.iat_min, a.iat_max, d.iat_mean, d.iat_std, d.pkt_rate, d.byte_rate, a.flags[0], a.flags[1],
                a.flags[2], a.flags[3], a.flags[4], a.flags[5], d.entropy, a.bursts, a.max_burst, d.mean_burst);
    }
    return fclose(f) == 0;
}

int main(int argc, char** argv) {
    std::string impl = "seq", input, out, save;
    size_t n = (size_t)1 << 20, flows = 0;
    double zipf = 1.0;
    uint64_t seed = 1;
    int threads = omp_get_max_threads(), parts = 10, reps = 1, warmup = 0;
    bool dyn = true, check = false, json = false, quiet = false;
    uint32_t gap = DEFAULT_GAP_US;
    GpuOpts go;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--input") input = next();
        else if (a == "--logn") n = (size_t)1 << atoi(next());
        else if (a == "--n") n = strtoull(next(), nullptr, 10);
        else if (a == "--flows") flows = strtoull(next(), nullptr, 10);
        else if (a == "--zipf") zipf = atof(next());
        else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
        else if (a == "--impl") impl = next();
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--parts") parts = atoi(next());
        else if (a == "--static") dyn = false;
        else if (a == "--tpb") go.tpb = atoi(next());
        else if (a == "--chunk") go.chunk = atoi(next());
        else if (a == "--pageable") go.pinned = false;
        else if (a == "--pinned-out") go.pinned_out = true;
        else if (a == "--gap") gap = (uint32_t)strtoul(next(), nullptr, 10);
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--check") check = true;
        else if (a == "--json") json = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--out") out = next();
        else if (a == "--save") save = next();
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (threads < 1) threads = 1;
    omp_set_num_threads(threads);

    PacketVec pk;
    if (!input.empty()) {
        if (!read_pkt(input, pk)) { fprintf(stderr, "cannot read %s\n", input.c_str()); return 1; }
        n = pk.size();
        flows = 0;
    } else {
        if (flows == 0) flows = n / 16 ? n / 16 : 1;
        pk = generate(n, flows, zipf, seed);
    }
    if (n == 0) { fprintf(stderr, "no packets\n"); return 1; }
    if (!save.empty()) {
        if (!write_pkt(save, pk.data(), n)) { fprintf(stderr, "cannot write %s\n", save.c_str()); return 1; }
        printf("wrote %zu packets to %s\n", n, save.c_str());
        return 0;
    }
    uint64_t in_bytes = 0;
    #pragma omp parallel for reduction(+ : in_bytes)
    for (size_t i = 0; i < n; i++) in_bytes += pk_len(pk[i].meta);
    double span = (pk[n - 1].ts - pk[0].ts) * 1e-6;

    bool is_gpu = impl.rfind("gpu", 0) == 0;
    if (!json) {
        printf("Packets: %zu (%.1f MB of traffic, %.3f s of capture)", n, in_bytes / 1e6, span);
        if (input.empty()) printf(", synthetic: %zu flows, Zipf exponent %.2f, seed %" PRIu64 "\n", flows, zipf, seed);
        else printf(", file %s\n", input.c_str());
        printf("Implementation: %s", impl.c_str());
        if (impl.rfind("omp", 0) == 0) printf(", %d threads", threads);
        if (is_gpu) printf(" on %s, %d threads per block", gpu_name().c_str(), go.tpb);
        printf("\n");
    }

    Result ref;
    if (check) ref = run_seq(pk.data(), n, gap);

    int rc = 0;
    Result res;
    for (int rep = -warmup; rep < reps; rep++) {
        res = run_impl(impl, pk.data(), n, gap, threads, parts, dyn, go);
        if (rep < 0) continue;
        uint64_t ck = checksum(res.recs);
        uint32_t top = 0;
        for (size_t j = 0; j < res.recs.size(); j++) if (res.recs[j].n > top) top = res.recs[j].n;
        size_t bad = 0;
        double max_rel = 0.0;
        std::string why;
        if (check) {
            bad = compare(ref, res, 1e-9, &max_rel, &why);
            if (bad) rc = 1;
        }
        if (json) {
            printf("RESULT {\"impl\":\"%s\",\"n\":%zu,\"flows_param\":%zu,\"zipf\":%.3f,\"seed\":%" PRIu64
                   ",\"input\":\"%s\",\"threads\":%d,\"parts\":%d,\"dynamic\":%d,\"tpb\":%d,\"chunk\":%d,\"pinned\":%d,\"pinned_out\":%d,"
                   "\"gap\":%u,\"rep\":%d,\"total_s\":%.9f,\"nflows\":%zu,\"top_flow_packets\":%u,\"bytes\":%" PRIu64
                   ",\"checksum\":\"%016" PRIx64 "\",\"checked\":%d,\"mismatch\":%zu,\"max_rel\":%.3e,\"phases\":{",
                   impl.c_str(), n, flows, zipf, seed, input.c_str(), threads, parts, dyn ? 1 : 0, go.tpb, go.chunk,
                   go.pinned ? 1 : 0, go.pinned_out ? 1 : 0, gap, rep, res.times.total, res.recs.size(), top, in_bytes, ck, check ? 1 : 0,
                   bad, max_rel);
            for (size_t k = 0; k < res.times.phase.size(); k++)
                printf("%s\"%s\":%.9f", k ? "," : "", res.times.phase[k].first.c_str(), res.times.phase[k].second);
            printf("}}\n");
        } else {
            printf("Run %d: %.4f s  (%.2f million packets/s)  flows %zu  checksum %016" PRIx64, rep + 1,
                   res.times.total, n / res.times.total / 1e6, res.recs.size(), ck);
            if (check) {
                if (bad) printf("  MISMATCH in %zu flows: %s", bad, why.c_str());
                else printf("  matches the sequential reference (max relative difference %.1e)", max_rel);
            }
            printf("\n   ");
            for (auto& ph : res.times.phase) printf(" %s %.4f s;", ph.first.c_str(), ph.second);
            printf("\n");
        }
        fflush(stdout);
    }

    if (!json && !quiet) {
        sort_by_key(res);
        uint32_t top = 0;
        size_t topj = 0;
        for (size_t j = 0; j < res.recs.size(); j++) if (res.recs[j].n > top) { top = res.recs[j].n; topj = j; }
        printf("Flows: %zu; the largest has %u packets (%.2f%% of all packets)\n", res.recs.size(), top,
               100.0 * top / n);
        printf("Largest flow and the first flows by key:\n");
        printf("  %-21s %-21s %5s %9s %12s %8s %8s %10s %10s %7s %7s %8s\n", "source", "destination", "proto",
               "packets", "bytes", "len mean", "len std", "iat mean", "iat std", "entropy", "bursts", "max burst");
        for (int q = -1; q < 4 && q < (int)res.recs.size(); q++) {
            size_t j = q < 0 ? topj : (size_t)q;
            const FlowRec& a = res.recs[j];
            const FlowDerived& d = res.derived[j];
            char s1[40], s2[40];
            snprintf(s1, sizeof s1, "%s:%u", ip_str(a.src).c_str(), a.ports >> 16);
            snprintf(s2, sizeof s2, "%s:%u", ip_str(a.dst).c_str(), a.ports & 0xFFFF);
            printf("  %-21s %-21s %5u %9u %12" PRIu64 " %8.1f %8.1f %10.1f %10.1f %7.3f %7u %8u\n", s1, s2, a.proto,
                   a.n, a.bytes, d.len_mean, d.len_std, d.iat_mean, d.iat_std, d.entropy, a.bursts, a.max_burst);
        }
    }
    if (!out.empty()) {
        sort_by_key(res);
        if (!write_csv(out, res)) { fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
        if (!json) printf("Wrote %zu flows to %s\n", res.recs.size(), out.c_str());
    }
    return rc;
}
