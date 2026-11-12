// Synthetic traffic generator, packet files, and result checks.
#include "flowfeat.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <omp.h>

#include <mutex>
#include <unordered_map>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace ff {

// ------------------------------------------------------------ buffer pool
namespace {
const size_t POOL_MIN = (size_t)1 << 20;       // smaller requests go to malloc
const size_t POOL_ALIGN = (size_t)1 << 21;     // 2 MiB: lets Linux use huge pages
const size_t POOL_MAX_CACHED = (size_t)12 << 30;
struct Pool {
    std::mutex mu;
    std::unordered_map<void*, size_t> size_of;          // every pool block, in use or free
    std::vector<std::pair<size_t, void*>> free_blocks;  // (capacity, block)
    size_t cached = 0;
};
Pool& pool() {
    static Pool* p = new Pool;  // never destroyed: buffers may outlive static destructors
    return *p;
}
}  // namespace

void* pool_get(size_t bytes) {
    if (bytes < POOL_MIN) return ::operator new(bytes);
    Pool& P = pool();
    size_t cap = (bytes + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN;
    {
        std::lock_guard<std::mutex> g(P.mu);
        size_t best = (size_t)-1;
        for (size_t i = 0; i < P.free_blocks.size(); i++)
            if (P.free_blocks[i].first >= cap && (best == (size_t)-1 || P.free_blocks[i].first < P.free_blocks[best].first))
                best = i;
        if (best != (size_t)-1 && P.free_blocks[best].first <= 2 * cap) {
            void* p = P.free_blocks[best].second;
            P.cached -= P.free_blocks[best].first;
            P.free_blocks[best] = P.free_blocks.back();
            P.free_blocks.pop_back();
            return p;
        }
    }
    void* p = aligned_alloc(POOL_ALIGN, cap);
    if (!p) throw std::bad_alloc();
#ifdef __linux__
    madvise(p, cap, MADV_HUGEPAGE);
#endif
    std::lock_guard<std::mutex> g(P.mu);
    P.size_of[p] = cap;
    return p;
}

void pool_put(void* p) {
    if (!p) return;
    Pool& P = pool();
    std::lock_guard<std::mutex> g(P.mu);
    auto it = P.size_of.find(p);
    if (it == P.size_of.end()) {  // a small block from operator new
        ::operator delete(p);
        return;
    }
    if (P.cached + it->second > POOL_MAX_CACHED) {
        P.size_of.erase(it);
        free(p);
        return;
    }
    P.free_blocks.emplace_back(it->second, p);
    P.cached += it->second;
}

void pool_trim() {
    Pool& P = pool();
    std::lock_guard<std::mutex> g(P.mu);
    for (auto& b : P.free_blocks) {
        P.size_of.erase(b.second);
        free(b.second);
    }
    P.free_blocks.clear();
    P.cached = 0;
}

// ------------------------------------------------------------ generator
// Counter-based random numbers: the i-th value depends only on (seed, i), so
// the generator can run in parallel and always gives the same packets.
static inline uint64_t rnd(uint64_t seed, uint64_t i) {
    return mix64(seed + 0x9e3779b97f4a7c15ULL * (i + 1));
}
static inline double u01(uint64_t x) { return (double)((x >> 11) + 1) * (1.0 / 9007199254740992.0); }

static const double MEAN_GAP_US = 2.0;  // 500,000 packets per second in total
static const uint32_t DPORTS[8] = {80, 443, 53, 22, 25, 123, 8080, 3306};

PacketVec generate(size_t n, size_t flows, double s, uint64_t seed, int threads) {
    if (threads <= 0) threads = omp_get_max_threads();
    if (flows < 1) flows = 1;
    PacketVec pk(n);

    // Flow keys by popularity rank.
    std::vector<Packet> key(flows);
    #pragma omp parallel for num_threads(threads)
    for (size_t r = 0; r < flows; r++) {
        uint64_t k = rnd(seed ^ 0x5bd1e995ULL, r), k2 = mix64(k);
        uint32_t proto = (k2 >> 8) % 10 < 8 ? 6u : 17u;
        key[r].src = 0x0A000000u | (uint32_t)(k & 0xFFFFFFu);
        key[r].dst = 0xC0A80000u | (uint32_t)((k >> 24) & 0xFFFFu);
        key[r].ports = ((1024u + (uint32_t)((k >> 40) % 64000u)) << 16) | DPORTS[k2 & 7];
        key[r].meta = proto;
        key[r].ts = 0;
    }
    // Zipf cumulative distribution: P(rank r) proportional to (r + 1)^-s.
    std::vector<double> cdf;
    if (s > 0.0) {
        cdf.resize(flows);
        double acc = 0.0;
        for (size_t r = 0; r < flows; r++) { acc += std::pow((double)(r + 1), -s); cdf[r] = acc; }
        for (size_t r = 0; r < flows; r++) cdf[r] /= acc;
        cdf[flows - 1] = 1.0;
    }
    // Poisson arrivals: exponential gaps, accumulated in time order.
    double t = 0.0;
    for (size_t i = 0; i < n; i++) {
        t += -std::log(u01(rnd(seed ^ 0x1234567ULL, i))) * MEAN_GAP_US;
        pk[i].ts = (uint32_t)t;
    }
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < n; i++) {
        double u = u01(rnd(seed, i));
        size_t r;
        if (s > 0.0) r = std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        else r = (size_t)(u * (double)flows);
        if (r >= flows) r = flows - 1;
        uint64_t x = rnd(seed ^ 0xabcdef01ULL, i);
        uint32_t proto = key[r].meta, len, fl = 0;
        uint32_t a = (uint32_t)(x % 100), b = (uint32_t)((x >> 8) & 0xFFFF), c = (uint32_t)(x >> 32);
        if (proto == 6) {
            if (a < 40) len = 40 + b % 41;            // acknowledgements and small segments
            else if (a < 80) len = 1500;              // full-size segments
            else len = 81 + b % 1419;
            fl = 0x10;                                // ACK
            if ((c & 3) == 0) fl |= 0x08;             // PSH
            if (((c >> 2) & 63) == 0) fl = 0x02;      // SYN
            else if (((c >> 8) & 63) == 0) fl |= 0x01;  // FIN
            if (((c >> 14) & 255) == 0) fl = 0x04;    // RST
            if (((c >> 22) & 1023) == 0) fl |= 0x20;  // URG
        } else {
            len = 60 + b % 1341;
        }
        pk[i].src = key[r].src; pk[i].dst = key[r].dst; pk[i].ports = key[r].ports;
        pk[i].meta = make_meta(proto, fl, len);
    }
    return pk;
}

// ------------------------------------------------------------ packet files
static const char PKT_MAGIC[8] = {'F', 'L', 'O', 'W', 'P', 'K', 'T', '1'};

bool write_pkt(const std::string& path, const Packet* pk, size_t n, uint64_t base_ts_us) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    uint64_t hdr[2] = {(uint64_t)n, base_ts_us};
    bool ok = fwrite(PKT_MAGIC, 1, 8, f) == 8 && fwrite(hdr, 8, 2, f) == 2 &&
              fwrite(pk, sizeof(Packet), n, f) == n;
    return fclose(f) == 0 && ok;
}

bool read_pkt(const std::string& path, PacketVec& pk, uint64_t* base_ts_us) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    uint64_t hdr[2];
    bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, PKT_MAGIC, 8) == 0 && fread(hdr, 8, 2, f) == 2;
    if (ok) {
        pk.resize(hdr[0]);
        ok = fread(pk.data(), sizeof(Packet), hdr[0], f) == hdr[0];
        if (base_ts_us) *base_ts_us = hdr[1];
    }
    fclose(f);
    return ok;
}

// ------------------------------------------------------------ checks
uint64_t checksum(const RecVec& recs) {
    uint64_t sum = 0;
    const size_t W = sizeof(FlowRec) / 8;
    #pragma omp parallel for reduction(+ : sum)
    for (size_t j = 0; j < recs.size(); j++) {
        uint64_t w[sizeof(FlowRec) / 8];
        memcpy(w, &recs[j], sizeof(FlowRec));
        uint64_t x = 0x243f6a8885a308d3ULL;
        for (size_t k = 0; k < W; k++) x = mix64(x ^ w[k]) + k;
        sum += x;
    }
    return sum;
}

static inline bool key_less(const FlowRec& a, const FlowRec& b) {
    if (a.src != b.src) return a.src < b.src;
    if (a.dst != b.dst) return a.dst < b.dst;
    if (a.ports != b.ports) return a.ports < b.ports;
    return a.proto < b.proto;
}

void sort_by_key(Result& r) {
    size_t f = r.recs.size();
    std::vector<uint32_t> idx(f);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return key_less(r.recs[a], r.recs[b]); });
    RecVec nr(f);
    for (size_t j = 0; j < f; j++) nr[j] = r.recs[idx[j]];
    r.recs.swap(nr);
    if (r.derived.size() == f) {
        DerVec nd(f);
        for (size_t j = 0; j < f; j++) nd[j] = r.derived[idx[j]];
        r.derived.swap(nd);
    }
}

size_t compare(const Result& ref_in, const Result& got_in, double rel_tol, double* max_rel,
               std::string* first_diff) {
    Result ref, got;
    ref.recs = ref_in.recs; ref.derived = ref_in.derived;
    got.recs = got_in.recs; got.derived = got_in.derived;
    sort_by_key(ref);
    sort_by_key(got);
    double mr = 0.0;
    size_t bad = 0;
    char msg[256];
    if (ref.recs.size() != got.recs.size()) {
        snprintf(msg, sizeof msg, "flow count differs: reference %zu, got %zu", ref.recs.size(), got.recs.size());
        if (first_diff) *first_diff = msg;
        size_t a = ref.recs.size(), b = got.recs.size();
        if (max_rel) *max_rel = 0.0;
        return a > b ? a - b : b - a;
    }
    const int ND = sizeof(FlowDerived) / sizeof(double);
    for (size_t j = 0; j < ref.recs.size(); j++) {
        bool diff = memcmp(&ref.recs[j], &got.recs[j], sizeof(FlowRec)) != 0;
        if (diff && bad == 0) {
            snprintf(msg, sizeof msg, "record %zu differs (n %u vs %u, bursts %u vs %u, max_burst %u vs %u)", j,
                     ref.recs[j].n, got.recs[j].n, ref.recs[j].bursts, got.recs[j].bursts,
                     ref.recs[j].max_burst, got.recs[j].max_burst);
            if (first_diff) *first_diff = msg;
        }
        double a[sizeof(FlowDerived) / sizeof(double)], b[sizeof(FlowDerived) / sizeof(double)];
        memcpy(a, &ref.derived[j], sizeof(FlowDerived));
        memcpy(b, &got.derived[j], sizeof(FlowDerived));
        for (int k = 0; k < ND; k++) {
            if (a[k] == b[k]) continue;
            double den = std::max(std::fabs(a[k]), std::fabs(b[k]));
            double rel = std::fabs(a[k] - b[k]) / den;
            if (!(rel <= rel_tol)) {
                if (!diff && bad == 0) {
                    snprintf(msg, sizeof msg, "derived feature %d of flow %zu differs: %.17g vs %.17g", k, j, a[k], b[k]);
                    if (first_diff) *first_diff = msg;
                }
                diff = true;
            }
            if (rel > mr) mr = rel;
        }
        if (diff) bad++;
    }
    if (max_rel) *max_rel = mr;
    return bad;
}

}  // namespace ff
