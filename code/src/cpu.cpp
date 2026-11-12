// CPU implementations: the sequential baseline and two OpenMP strategies.
#include "flowfeat.h"
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace ff {

// Add one packet to a table. Every CPU implementation uses this same kernel.
static inline void add_packet(FlowTable& t, const Packet& p, uint32_t gap) {
    bool is_new;
    uint32_t proto = pk_proto(p.meta);
    uint32_t i = t.find_or_insert(p.src, p.dst, p.ports, proto,
                                  flow_hash(p.src, p.dst, p.ports, proto), &is_new);
    if (is_new) rec_init(t.recs[i], p);
    else rec_update(t.recs[i], p, gap);
}

// ------------------------------------------------------------ sequential
Result run_seq(const Packet* pk, size_t n, uint32_t gap) {
    Result res;
    double t0 = now_s();
    FlowTable t(1 << 15);
    for (size_t i = 0; i < n; i++) add_packet(t, pk[i], gap);
    double t1 = now_s();
    res.recs.swap(t.recs);
    size_t f = res.recs.size();
    res.derived.resize(f);
    for (size_t j = 0; j < f; j++) {
        rec_finish(res.recs[j]);
        res.derived[j] = derive(res.recs[j]);
    }
    double t2 = now_s();
    res.times.add("aggregate", t1 - t0);
    res.times.add("finalize", t2 - t1);
    res.times.total = t2 - t0;
    return res;
}

// Concatenate per-task outputs into one result, in parallel.
static void concat(std::vector<RecVec>& recs, std::vector<DerVec>& der, Result& res, int threads) {
    size_t k = recs.size();
    std::vector<size_t> off(k + 1, 0);
    for (size_t i = 0; i < k; i++) off[i + 1] = off[i] + recs[i].size();
    res.recs.resize(off[k]);
    res.derived.resize(off[k]);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < k; i++) {
        std::copy(recs[i].begin(), recs[i].end(), res.recs.begin() + off[i]);
        std::copy(der[i].begin(), der[i].end(), res.derived.begin() + off[i]);
    }
}

// ------------------------------------------------------------ OpenMP 1
// Each thread aggregates one contiguous (in time) chunk of the packets into a
// private table. The partial records are then merged in chunk order with the
// associative rec_merge. The merge is parallel too: the hash space is cut into
// ranges and each range is merged independently.
Result run_omp_merge(const Packet* pk, size_t n, uint32_t gap, int threads) {
    Result res;
    const int T = threads;
    int rb = 6;
    while ((1 << rb) < 8 * T) rb++;
    const size_t R = (size_t)1 << rb;
    const int shift = 64 - rb;

    double t0 = now_s();
    std::vector<FlowTable> local(T);
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        size_t lo = n * (size_t)t / T, hi = n * (size_t)(t + 1) / T;
        FlowTable L(1 << 12);
        for (size_t i = lo; i < hi; i++) add_packet(L, pk[i], gap);
        local[t] = std::move(L);
    }
    double t1 = now_s();

    // Bucket every local record by hash range (counting sort of indices).
    std::vector<std::vector<uint32_t>> bstart(T), bidx(T);
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        const RecVec& rv = local[t].recs;
        std::vector<uint32_t>& st = bstart[t];
        std::vector<uint32_t>& ix = bidx[t];
        st.assign(R + 1, 0);
        ix.resize(rv.size());
        for (size_t j = 0; j < rv.size(); j++)
            st[(flow_hash(rv[j].src, rv[j].dst, rv[j].ports, rv[j].proto) >> shift) + 1]++;
        for (size_t r = 0; r < R; r++) st[r + 1] += st[r];
        std::vector<uint32_t> pos(st.begin(), st.end() - 1);
        for (size_t j = 0; j < rv.size(); j++)
            ix[pos[flow_hash(rv[j].src, rv[j].dst, rv[j].ports, rv[j].proto) >> shift]++] = (uint32_t)j;
    }
    double t2 = now_s();

    std::vector<RecVec> outs(R);
    std::vector<DerVec> ders(R);
    #pragma omp parallel num_threads(T)
    {
        FlowTable M;
        #pragma omp for schedule(dynamic, 1)
        for (size_t r = 0; r < R; r++) {
            size_t expect = 0;
            for (int t = 0; t < T; t++)
                expect = std::max(expect, (size_t)(bstart[t][r + 1] - bstart[t][r]));
            M.reset(expect);
            for (int t = 0; t < T; t++) {  // chunk order = time order
                const RecVec& rv = local[t].recs;
                for (uint32_t q = bstart[t][r]; q < bstart[t][r + 1]; q++) {
                    const FlowRec& b = rv[bidx[t][q]];
                    bool is_new;
                    uint32_t i = M.find_or_insert(b.src, b.dst, b.ports, b.proto,
                                                  flow_hash(b.src, b.dst, b.ports, b.proto), &is_new);
                    if (is_new) M.recs[i] = b;
                    else rec_merge(M.recs[i], b, gap);
                }
            }
            size_t f = M.recs.size();
            ders[r].resize(f);
            for (size_t j = 0; j < f; j++) {
                rec_finish(M.recs[j]);
                ders[r][j] = derive(M.recs[j]);
            }
            outs[r].swap(M.recs);
        }
    }
    double t3 = now_s();
    concat(outs, ders, res, T);
    double t4 = now_s();
    res.times.add("local_aggregate", t1 - t0);
    res.times.add("bucket", t2 - t1);
    res.times.add("merge_finalize", t3 - t2);
    res.times.add("concat", t4 - t3);
    res.times.total = t4 - t0;
    return res;
}

// ------------------------------------------------------------ OpenMP 2
// Packets are first partitioned by the top bits of the flow hash, so all the
// packets of a flow land in the same partition, still in time order. Each
// partition is then aggregated by one thread with a private table: there is
// no shared state and nothing to merge.
Result run_omp_part(const Packet* pk, size_t n, uint32_t gap, int threads, int lp, bool dynamic) {
    Result res;
    const int T = threads;
    const size_t P = (size_t)1 << lp;
    const int shift = 64 - lp;

    double t0 = now_s();
    std::vector<size_t> cnt((size_t)T * P, 0);
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        size_t lo = n * (size_t)t / T, hi = n * (size_t)(t + 1) / T;
        std::vector<size_t> c(P, 0);
        for (size_t i = lo; i < hi; i++) c[pk_hash(pk[i]) >> shift]++;
        std::copy(c.begin(), c.end(), cnt.begin() + (size_t)t * P);
    }
    std::vector<size_t> pstart(P + 1, 0);
    size_t pos = 0;
    for (size_t p = 0; p < P; p++) {  // thread order inside a partition keeps time order
        pstart[p] = pos;
        for (int t = 0; t < T; t++) {
            size_t c = cnt[(size_t)t * P + p];
            cnt[(size_t)t * P + p] = pos;
            pos += c;
        }
    }
    pstart[P] = pos;
    PacketVec buf(n);
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        size_t lo = n * (size_t)t / T, hi = n * (size_t)(t + 1) / T;
        std::vector<size_t> o(cnt.begin() + (size_t)t * P, cnt.begin() + (size_t)(t + 1) * P);
        for (size_t i = lo; i < hi; i++) buf[o[pk_hash(pk[i]) >> shift]++] = pk[i];
    }
    double t1 = now_s();

    // Largest partitions first, so the dynamic schedule can balance the rest.
    std::vector<uint32_t> order(P);
    std::iota(order.begin(), order.end(), 0u);
    if (dynamic)
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return pstart[a + 1] - pstart[a] > pstart[b + 1] - pstart[b];
        });
    std::vector<RecVec> outs(T);
    std::vector<DerVec> ders(T);
    omp_set_schedule(dynamic ? omp_sched_dynamic : omp_sched_static, dynamic ? 1 : 0);
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        FlowTable M;
        RecVec out;
        DerVec der;
        #pragma omp for schedule(runtime) nowait
        for (size_t q = 0; q < P; q++) {
            size_t p = order[q];
            size_t lo = pstart[p], hi = pstart[p + 1];
            if (lo == hi) continue;
            M.reset((hi - lo) / 8);
            for (size_t i = lo; i < hi; i++) add_packet(M, buf[i], gap);
            for (size_t j = 0; j < M.recs.size(); j++) {
                rec_finish(M.recs[j]);
                out.push_back(M.recs[j]);
                der.push_back(derive(M.recs[j]));
            }
        }
        outs[t].swap(out);
        ders[t].swap(der);
    }
    double t2 = now_s();
    concat(outs, ders, res, T);
    double t3 = now_s();
    res.times.add("partition", t1 - t0);
    res.times.add("aggregate_finalize", t2 - t1);
    res.times.add("concat", t3 - t2);
    res.times.total = t3 - t0;
    return res;
}

}  // namespace ff
