// C interface for Python (ctypes): used by the MPI4Py driver and the tests.
#include "flowfeat.h"
#include <cstdio>
#include <cstring>
#include <omp.h>

using namespace ff;

namespace {
struct Handle {
    Result res;
};
Result run_named(int impl, const Packet* pk, size_t n, uint32_t gap, int threads) {
    if (threads > 0) omp_set_num_threads(threads);
    else threads = omp_get_max_threads();
    switch (impl) {
        case 0: return run_seq(pk, n, gap);
        case 1: return run_omp_merge(pk, n, gap, threads);
        case 2: return run_omp_part(pk, n, gap, threads);
        default: {
            GpuOpts g;
            g.variant = impl - 3;  // 3 = gpu-flow, 4 = gpu-atomic, 5 = gpu-chunk
            return run_gpu(pk, n, gap, g);
        }
    }
}
}  // namespace

extern "C" {

int ff_packet_size() { return (int)sizeof(Packet); }
int ff_rec_size() { return (int)sizeof(FlowRec); }
int ff_derived_size() { return (int)sizeof(FlowDerived); }
int ff_gpu_available() { return gpu_available() ? 1 : 0; }

// Synthetic packets into a caller-provided buffer of n * 20 bytes.
void ff_generate(void* out, uint64_t n, uint64_t flows, double zipf, uint64_t seed) {
    PacketVec pk = generate(n, flows, zipf, seed);
    memcpy(out, pk.data(), n * sizeof(Packet));
}

// Number of packets in a .pkt file (0 on error), then read them.
uint64_t ff_pkt_count(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    char magic[8];
    uint64_t hdr[2] = {0, 0};
    bool ok = fread(magic, 1, 8, f) == 8 && fread(hdr, 8, 2, f) == 2;
    fclose(f);
    return ok ? hdr[0] : 0;
}
int ff_pkt_read(const char* path, void* out, uint64_t n) {
    PacketVec pk;
    if (!read_pkt(path, pk) || pk.size() != n) return 1;
    memcpy(out, pk.data(), n * sizeof(Packet));
    return 0;
}

// Extract features. Returns a handle; *nflows, *seconds and *check are filled.
void* ff_extract(const void* packets, uint64_t n, int impl, int threads, uint32_t gap, uint64_t* nflows,
                 double* seconds, uint64_t* check) {
    Handle* h = new Handle;
    h->res = run_named(impl, (const Packet*)packets, n, gap, threads);
    if (nflows) *nflows = h->res.recs.size();
    if (seconds) *seconds = h->res.times.total;
    if (check) *check = checksum(h->res.recs);
    return h;
}
// Copy the records (176 bytes each) and derived features (9 doubles each),
// sorted by flow key if sorted != 0.
void ff_copy(void* handle, void* recs, void* derived, int sorted) {
    Handle* h = (Handle*)handle;
    if (sorted) sort_by_key(h->res);
    if (recs) memcpy(recs, h->res.recs.data(), h->res.recs.size() * sizeof(FlowRec));
    if (derived) memcpy(derived, h->res.derived.data(), h->res.derived.size() * sizeof(FlowDerived));
}
void ff_free(void* handle) { delete (Handle*)handle; }

// One call for a whole batch file: read, extract, return summary numbers.
// times[0] = read seconds, times[1] = extraction seconds.
int ff_process_file(const char* path, int impl, int threads, uint32_t gap, uint64_t* npackets, uint64_t* nflows,
                    uint64_t* check, double* times) {
    double t0 = now_s();
    PacketVec pk;
    if (!read_pkt(path, pk)) return 1;
    double t1 = now_s();
    Result r = run_named(impl, pk.data(), pk.size(), gap, threads);
    double t2 = now_s();
    *npackets = pk.size();
    *nflows = r.recs.size();
    *check = checksum(r.recs);
    times[0] = t1 - t0;
    times[1] = t2 - t1;
    return 0;
}
}
