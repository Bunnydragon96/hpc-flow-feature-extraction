"""ctypes binding of libflowfeat.so (build it with `make` in the code folder)."""
import ctypes as C
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "..", "libflowfeat.so")

NFLAGS, NBINS = 6, 16
PACKET = np.dtype([("ts", "<u4"), ("src", "<u4"), ("dst", "<u4"), ("ports", "<u4"), ("meta", "<u4")])
REC = np.dtype([
    ("bytes", "<u8"), ("sum_len2", "<u8"), ("sum_iat2", "<u8"),
    ("src", "<u4"), ("dst", "<u4"), ("ports", "<u4"), ("proto", "<u4"),
    ("n", "<u4"), ("len_min", "<u4"), ("len_max", "<u4"), ("first_ts", "<u4"), ("last_ts", "<u4"),
    ("iat_min", "<u4"), ("iat_max", "<u4"), ("bursts", "<u4"), ("max_burst", "<u4"),
    ("head_burst", "<u4"), ("tail_burst", "<u4"),
    ("flags", "<u4", (NFLAGS,)), ("hist", "<u4", (NBINS,)), ("pad", "<u4"),
])
DERIVED_NAMES = ["duration_s", "len_mean", "len_std", "iat_mean", "iat_std", "pkt_rate", "byte_rate",
                 "entropy", "mean_burst"]
DERIVED = np.dtype([(k, "<f8") for k in DERIVED_NAMES])
IMPLS = {"seq": 0, "omp-merge": 1, "omp-part": 2, "gpu-flow": 3, "gpu-atomic": 4, "gpu-chunk": 5}
DEFAULT_GAP_US = 1000

_lib = None


def lib():
    global _lib
    if _lib is None:
        L = C.CDLL(LIB_PATH)
        L.ff_generate.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_double, C.c_uint64]
        L.ff_generate.restype = None
        L.ff_pkt_count.argtypes = [C.c_char_p]
        L.ff_pkt_count.restype = C.c_uint64
        L.ff_pkt_read.argtypes = [C.c_char_p, C.c_void_p, C.c_uint64]
        L.ff_extract.argtypes = [C.c_void_p, C.c_uint64, C.c_int, C.c_int, C.c_uint32,
                                 C.POINTER(C.c_uint64), C.POINTER(C.c_double), C.POINTER(C.c_uint64)]
        L.ff_extract.restype = C.c_void_p
        L.ff_copy.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
        L.ff_copy.restype = None
        L.ff_free.argtypes = [C.c_void_p]
        L.ff_free.restype = None
        L.ff_process_file.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_uint32, C.POINTER(C.c_uint64),
                                      C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.POINTER(C.c_double)]
        assert L.ff_packet_size() == PACKET.itemsize and L.ff_rec_size() == REC.itemsize
        assert L.ff_derived_size() == DERIVED.itemsize
        _lib = L
    return _lib


def gpu_available():
    return bool(lib().ff_gpu_available())


def generate(n, flows, zipf, seed=1):
    """Synthetic packets as a NumPy structured array."""
    pk = np.empty(n, dtype=PACKET)
    lib().ff_generate(pk.ctypes.data, n, flows, zipf, seed)
    return pk


def read_pkt(path):
    n = lib().ff_pkt_count(path.encode())
    pk = np.empty(n, dtype=PACKET)
    if n == 0 or lib().ff_pkt_read(path.encode(), pk.ctypes.data, n) != 0:
        raise IOError("cannot read " + path)
    return pk


def extract(pk, impl="seq", threads=0, gap=DEFAULT_GAP_US, sort=True):
    """Returns (records, derived, seconds, checksum)."""
    nf, sec, chk = C.c_uint64(), C.c_double(), C.c_uint64()
    pk = np.ascontiguousarray(pk)
    h = lib().ff_extract(pk.ctypes.data, len(pk), IMPLS[impl], threads, gap, C.byref(nf), C.byref(sec), C.byref(chk))
    recs = np.empty(nf.value, dtype=REC)
    der = np.empty(nf.value, dtype=DERIVED)
    lib().ff_copy(h, recs.ctypes.data, der.ctypes.data, 1 if sort else 0)
    lib().ff_free(h)
    return recs, der, sec.value, chk.value


def process_file(path, impl="seq", threads=1, gap=DEFAULT_GAP_US):
    """Read a .pkt batch and extract its features inside the library.
    Returns a dict with packets, flows, checksum, read_s, compute_s."""
    npk, nf, chk = C.c_uint64(), C.c_uint64(), C.c_uint64()
    times = (C.c_double * 2)()
    rc = lib().ff_process_file(path.encode(), IMPLS[impl], threads, gap, C.byref(npk), C.byref(nf), C.byref(chk), times)
    if rc != 0:
        raise IOError("cannot read " + path)
    return {"packets": npk.value, "flows": nf.value, "checksum": chk.value, "read_s": times[0], "compute_s": times[1]}
