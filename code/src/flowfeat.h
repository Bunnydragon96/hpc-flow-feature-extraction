// Host-side interface: packet sources, the implementations, result checks.
#pragma once
#include "common.h"
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace ff {

// ---------------------------------------------------------------- timing
inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
// Named phase times of one run, in seconds, in execution order.
struct Times {
    std::vector<std::pair<std::string, double>> phase;
    double total = 0.0;
    void add(const std::string& name, double s) { phase.emplace_back(name, s); }
};

// Large buffers (tables, partition buffers, outputs) are recycled between
// runs instead of being returned to the operating system, as a pipeline that
// processes one batch after another would do. Without this, every run pays
// for fresh page faults on hundreds of megabytes. Thread-safe.
void* pool_get(size_t bytes);
void pool_put(void* p);
void pool_trim();   // give all cached blocks back to the operating system

// Allocator on top of the pool. It also leaves plain structs uninitialised, so
// large arrays are not zero-filled by one thread before parallel code writes them.
template <class T>
struct NoInit {
    typedef T value_type;
    NoInit() = default;
    template <class U> NoInit(const NoInit<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(pool_get(n * sizeof(T))); }
    void deallocate(T* p, size_t) { pool_put(p); }
    template <class U> void construct(U* p) { ::new ((void*)p) U; }
    template <class U, class A0, class... A> void construct(U* p, A0&& a0, A&&... a) {
        ::new ((void*)p) U(std::forward<A0>(a0), std::forward<A>(a)...);
    }
    template <class U> bool operator==(const NoInit<U>&) const { return true; }
    template <class U> bool operator!=(const NoInit<U>&) const { return false; }
};
typedef std::vector<FlowRec, NoInit<FlowRec>> RecVec;
typedef std::vector<FlowDerived, NoInit<FlowDerived>> DerVec;
typedef std::vector<Packet, NoInit<Packet>> PacketVec;

struct Result {
    RecVec recs;      // one record per flow, arbitrary order
    DerVec derived;   // same order as recs
    Times times;
};

// ---------------------------------------------------------------- hash table
// Open addressing with linear probing. Slots hold an index into recs.
class FlowTable {
public:
    RecVec recs;
    explicit FlowTable(size_t expected = 1024) { reset(expected); }
    void reset(size_t expected) {
        size_t cap = 1024;
        while (cap < 2 * expected) cap <<= 1;
        slots_.assign(cap, EMPTY);
        mask_ = cap - 1;
        recs.clear();
    }
    // Index of the record of the flow with this key (hash h). If *is_new, the
    // caller must initialise the record (including its key) right away.
    inline uint32_t find_or_insert(uint32_t src, uint32_t dst, uint32_t ports, uint32_t proto,
                                   uint64_t h, bool* is_new) {
        size_t s = (size_t)h & mask_;
        for (;;) {
            uint32_t idx = slots_[s];
            if (idx == EMPTY) break;
            const FlowRec& r = recs[idx];
            if (r.src == src && r.dst == dst && r.ports == ports && r.proto == proto) {
                *is_new = false;
                return idx;
            }
            s = (s + 1) & mask_;
        }
        if (2 * (recs.size() + 1) > slots_.size()) {  // keep the load factor <= 0.5
            grow();
            s = (size_t)h & mask_;
            while (slots_[s] != EMPTY) s = (s + 1) & mask_;
        }
        uint32_t idx = (uint32_t)recs.size();
        slots_[s] = idx;
        recs.emplace_back();
        *is_new = true;
        return idx;
    }
    size_t capacity() const { return slots_.size(); }

private:
    static constexpr uint32_t EMPTY = 0xFFFFFFFFu;
    std::vector<uint32_t, NoInit<uint32_t>> slots_;
    size_t mask_ = 0;
    void grow() {  // double the table and re-insert every record
        size_t cap = slots_.size() * 2;
        slots_.assign(cap, EMPTY);
        mask_ = cap - 1;
        for (uint32_t i = 0; i < (uint32_t)recs.size(); i++) {
            const FlowRec& r = recs[i];
            size_t s = (size_t)flow_hash(r.src, r.dst, r.ports, r.proto) & mask_;
            while (slots_[s] != EMPTY) s = (s + 1) & mask_;
            slots_[s] = i;
        }
    }
};

// ---------------------------------------------------------------- inputs
// Synthetic traffic: n packets over `flows` flows whose popularity follows a
// Zipf law with exponent s (s = 0 is uniform). Deterministic for a given seed.
PacketVec generate(size_t n, size_t flows, double zipf_s, uint64_t seed, int threads = 0);

// Binary packet files (".pkt"): 24-byte header + packets.
bool write_pkt(const std::string& path, const Packet* pk, size_t n, uint64_t base_ts_us = 0);
bool read_pkt(const std::string& path, PacketVec& pk, uint64_t* base_ts_us = nullptr);

// ---------------------------------------------------------------- implementations
Result run_seq(const Packet* pk, size_t n, uint32_t gap);
// Thread-local tables over contiguous time chunks, then a parallel merge.
Result run_omp_merge(const Packet* pk, size_t n, uint32_t gap, int threads);
// Partition packets by flow hash, then one private table per partition.
Result run_omp_part(const Packet* pk, size_t n, uint32_t gap, int threads, int log2_parts = 10,
                    bool dynamic = true);

struct GpuOpts {
    int variant = 2;      // 0 = one thread per flow, 1 = one thread per packet (atomics), 2 = chunked
    int tpb = 256;        // threads per block
    int chunk = 8;        // packets per thread in the chunked variant (1..255)
    bool pinned = true;       // packets are staged in page-locked host memory before the upload
    bool pinned_out = false;  // download into page-locked memory, then copy (default: download directly)
};
bool gpu_available();
std::string gpu_name();
Result run_gpu(const Packet* pk, size_t n, uint32_t gap, const GpuOpts& opt);

// ---------------------------------------------------------------- checks
// Order-independent checksum of the records (sum of per-record hashes).
uint64_t checksum(const RecVec& recs);
// Compare two results flow by flow. Returns the number of flows that differ
// (records must match exactly; derived features within rel_tol). max_rel
// receives the largest relative difference seen in a derived feature.
size_t compare(const Result& ref, const Result& got, double rel_tol, double* max_rel,
               std::string* first_diff);
// Sort records (and derived) by flow key.
void sort_by_key(Result& r);

}  // namespace ff
