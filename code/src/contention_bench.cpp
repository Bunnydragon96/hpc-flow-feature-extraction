// contention_bench: what happens when threads update ONE shared flow table.
//
// The micro-benchmark aggregates only the order-independent features (packet
// and byte counts, length extremes, sum of squares, flag counts, length
// histogram), because those are the ones a shared table can compute at all.
// Four strategies are timed on the same packets:
//   seq     one thread, private table (reference)
//   lock    shared table, one spinlock per flow
//   atomic  shared table, every field updated with an atomic instruction
//   local   one private table per thread, merged by one thread at the end
// Each thread processes a contiguous range of the packets. Only the update
// loop (plus the merge for "local") is timed; tables are allocated before.
//
//   contention_bench --logn 24 --flows 1048576 --zipf 1.2 --threads 8 --reps 5 --json
#include "flowfeat.h"
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <omp.h>
#include <string>

using namespace ff;

static const uint32_t EMPTY = 0xFFFFFFFFu;

struct CRec {  // plain record, 2 cache lines
    uint64_t bytes, sum_len2;
    uint32_t n, len_min, len_max;
    uint32_t flags[NFLAGS];
    uint32_t hist[NBINS];
    uint32_t owner;  // index of the packet that defines the key, EMPTY if unused
    uint32_t lock;
    uint32_t pad;
};
static_assert(sizeof(CRec) == 128, "CRec must be 128 bytes");

static inline void upd_plain(CRec& r, uint32_t meta) {
    uint32_t len = pk_len(meta), fl = pk_flags(meta);
    r.n++;
    r.bytes += len;
    r.sum_len2 += (uint64_t)len * len;
    if (len < r.len_min) r.len_min = len;
    if (len > r.len_max) r.len_max = len;
    for (int k = 0; k < NFLAGS; k++) r.flags[k] += (fl >> k) & 1u;
    r.hist[len_bin(len)]++;
}
static inline void upd_atomic(CRec& r, uint32_t meta) {
    uint32_t len = pk_len(meta), fl = pk_flags(meta);
    __atomic_fetch_add(&r.n, 1u, __ATOMIC_RELAXED);
    __atomic_fetch_add(&r.bytes, (uint64_t)len, __ATOMIC_RELAXED);
    __atomic_fetch_add(&r.sum_len2, (uint64_t)len * len, __ATOMIC_RELAXED);
    uint32_t cur = __atomic_load_n(&r.len_min, __ATOMIC_RELAXED);
    while (len < cur && !__atomic_compare_exchange_n(&r.len_min, &cur, len, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    cur = __atomic_load_n(&r.len_max, __ATOMIC_RELAXED);
    while (len > cur && !__atomic_compare_exchange_n(&r.len_max, &cur, len, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    for (int k = 0; k < NFLAGS; k++)
        if ((fl >> k) & 1u) __atomic_fetch_add(&r.flags[k], 1u, __ATOMIC_RELAXED);
    __atomic_fetch_add(&r.hist[len_bin(len)], 1u, __ATOMIC_RELAXED);
}
static inline void upd_lock(CRec& r, uint32_t meta) {
    for (;;) {  // test-and-test-and-set spinlock
        if (__atomic_load_n(&r.lock, __ATOMIC_RELAXED) == 0 &&
            __atomic_exchange_n(&r.lock, 1u, __ATOMIC_ACQUIRE) == 0) break;
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
    upd_plain(r, meta);
    __atomic_store_n(&r.lock, 0u, __ATOMIC_RELEASE);
}

struct Table {
    CRec* rec = nullptr;
    size_t cap = 0, mask = 0;
    std::vector<uint32_t> used;  // slots claimed through find_private
    void alloc(size_t flows) {
        cap = 1024;
        while (cap < 4 * flows) cap <<= 1;
        mask = cap - 1;
        rec = (CRec*)aligned_alloc(64, cap * sizeof(CRec));
        clear();
    }
    void clear() {
        used.clear();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cap; i++) {
            memset((void*)&rec[i], 0, sizeof(CRec));
            rec[i].len_min = 0xFFFFFFFFu;
            rec[i].owner = EMPTY;
        }
    }
    // Private use: find or claim the slot of packet i.
    inline CRec& find_private(const Packet* pk, uint32_t i) {
        size_t s = (size_t)pk_hash(pk[i]) & mask;
        for (;;) {
            uint32_t o = rec[s].owner;
            if (o == EMPTY) { rec[s].owner = i; used.push_back((uint32_t)s); return rec[s]; }
            if (same_flow(pk[o], pk[i])) return rec[s];
            s = (s + 1) & mask;
        }
    }
    // Shared use: the slot is claimed with one compare-and-swap.
    inline CRec& find_shared(const Packet* pk, uint32_t i) {
        size_t s = (size_t)pk_hash(pk[i]) & mask;
        for (;;) {
            uint32_t o = __atomic_load_n(&rec[s].owner, __ATOMIC_ACQUIRE);
            if (o == EMPTY) {
                uint32_t expect = EMPTY;
                if (__atomic_compare_exchange_n(&rec[s].owner, &expect, i, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    return rec[s];
                o = expect;
            }
            if (same_flow(pk[o], pk[i])) return rec[s];
            s = (s + 1) & mask;
        }
    }
    uint64_t checksum(const Packet* pk, size_t* nflows) const {
        uint64_t sum = 0;
        size_t nf = 0;
        #pragma omp parallel for reduction(+ : sum, nf)
        for (size_t i = 0; i < cap; i++) {
            const CRec& r = rec[i];
            if (r.owner == EMPTY) continue;
            nf++;
            uint64_t x = pk_hash(pk[r.owner]);
            x = mix64(x ^ r.n) + mix64(r.bytes) + mix64(r.sum_len2 ^ ((uint64_t)r.len_min << 32 | r.len_max));
            for (int k = 0; k < NFLAGS; k++) x = mix64(x ^ r.flags[k]);
            for (int k = 0; k < NBINS; k++) x = mix64(x ^ r.hist[k]);
            sum += x;
        }
        *nflows = nf;
        return sum;
    }
};

int main(int argc, char** argv) {
    size_t n = (size_t)1 << 22, flows = 0;
    double zipf = 1.0;
    uint64_t seed = 1;
    int threads = omp_get_max_threads(), reps = 3;
    bool json = false;
    std::string only;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { if (i + 1 >= argc) exit(2); return argv[++i]; };
        if (a == "--logn") n = (size_t)1 << atoi(next());
        else if (a == "--flows") flows = strtoull(next(), nullptr, 10);
        else if (a == "--zipf") zipf = atof(next());
        else if (a == "--seed") seed = strtoull(next(), nullptr, 10);
        else if (a == "--threads") threads = atoi(next());
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--only") only = next();
        else if (a == "--json") json = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (flows == 0) flows = n / 16;
    omp_set_num_threads(threads);
    PacketVec pkv = generate(n, flows, zipf, seed);
    const Packet* pk = pkv.data();
    const int T = threads;

    Table shared;
    shared.alloc(flows);
    std::vector<Table> local(T);
    for (int t = 0; t < T; t++) local[t].alloc(flows);

    uint64_t ref_sum = 0;
    size_t ref_flows = 0;
    int rc = 0;
    const char* names[4] = {"seq", "lock", "atomic", "local"};
    if (!json) printf("%zu packets, %zu flows, Zipf exponent %.2f, %d threads\n", n, flows, zipf, T);
    for (int v = 0; v < 4; v++) {
        if (v > 0 && !only.empty() && only != names[v]) continue;
        for (int rep = 0; rep < (v == 0 ? 1 : 0) + reps; rep++) {
            shared.clear();
            if (v == 3) for (int t = 0; t < T; t++) local[t].clear();
            double t0 = now_s(), tm = 0.0;
            if (v == 0) {
                for (size_t i = 0; i < n; i++) upd_plain(shared.find_private(pk, (uint32_t)i), pk[i].meta);
            } else if (v == 1 || v == 2) {
                #pragma omp parallel num_threads(T)
                {
                    int t = omp_get_thread_num();
                    size_t lo = n * (size_t)t / T, hi = n * (size_t)(t + 1) / T;
                    if (v == 1) for (size_t i = lo; i < hi; i++) upd_lock(shared.find_shared(pk, (uint32_t)i), pk[i].meta);
                    else for (size_t i = lo; i < hi; i++) upd_atomic(shared.find_shared(pk, (uint32_t)i), pk[i].meta);
                }
            } else {
                #pragma omp parallel num_threads(T)
                {
                    int t = omp_get_thread_num();
                    size_t lo = n * (size_t)t / T, hi = n * (size_t)(t + 1) / T;
                    Table& L = local[t];
                    for (size_t i = lo; i < hi; i++) upd_plain(L.find_private(pk, (uint32_t)i), pk[i].meta);
                }
                tm = now_s();
                for (int t = 0; t < T; t++) {  // merge by one thread
                    const Table& L = local[t];
                    for (uint32_t s : L.used) {
                        const CRec& b = L.rec[s];
                        CRec& a = shared.find_private(pk, b.owner);
                        a.n += b.n; a.bytes += b.bytes; a.sum_len2 += b.sum_len2;
                        if (b.len_min < a.len_min) a.len_min = b.len_min;
                        if (b.len_max > a.len_max) a.len_max = b.len_max;
                        for (int k = 0; k < NFLAGS; k++) a.flags[k] += b.flags[k];
                        for (int k = 0; k < NBINS; k++) a.hist[k] += b.hist[k];
                    }
                }
            }
            double t1 = now_s();
            size_t nf;
            uint64_t sum = shared.checksum(pk, &nf);
            if (v == 0 && rep == 0) { ref_sum = sum; ref_flows = nf; continue; }  // warm-up and reference
            bool ok = sum == ref_sum && nf == ref_flows;
            if (!ok) rc = 1;
            if (json)
                printf("RESULT {\"bench\":\"contention\",\"variant\":\"%s\",\"n\":%zu,\"flows_param\":%zu,\"zipf\":%.3f,"
                       "\"threads\":%d,\"rep\":%d,\"total_s\":%.9f,\"merge_s\":%.9f,\"nflows\":%zu,\"ok\":%d}\n",
                       names[v], n, flows, zipf, v == 0 ? 1 : T, rep - (v == 0 ? 1 : 0), t1 - t0, tm > 0 ? t1 - tm : 0.0, nf, ok ? 1 : 0);
            else
                printf("  %-6s %.4f s  %7.2f million packets/s  flows %zu  %s\n", names[v], t1 - t0, n / (t1 - t0) / 1e6,
                       nf, ok ? "matches seq" : "MISMATCH");
            fflush(stdout);
        }
    }
    return rc;
}
