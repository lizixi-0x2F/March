"""
march._core — low-level ctypes binding and MarchCache
"""
import ctypes
import os
import time

def _find_lib():
    """Locate libmarch.so relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "libmarch.so"),
        os.path.join(here, "..", "march", "libmarch.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "libmarch.so not found. Run `make` inside the march directory first."
    )


class MarchCache:
    """
    Trie-based KV cache with prefix sharing.

    Parameters
    ----------
    page_size : int
        Bytes per page (should match your KV tensor slice size).
    max_pages : int
        Maximum number of pages in the pool.
    lib_path : str | None
        Override path to libmarch.so. Auto-detected if None.
    """

    def __init__(self, page_size: int = 256, max_pages: int = 64, lib_path=None):
        path = lib_path or _find_lib()
        self._lib = ctypes.CDLL(path)
        self._setup_signatures()
        self._ctx = self._lib.march_create(page_size, max_pages)
        if not self._ctx:
            raise RuntimeError("march_create returned NULL")
        self.page_size = page_size
        self.max_pages = max_pages
        # stats counters
        self._inserts = 0
        self._queries = 0
        self._hits = 0          # queries where matched_tokens > 0
        self._total_matched = 0
        self._total_queried = 0
        self._start_time = time.monotonic()

    def _setup_signatures(self):
        lib = self._lib
        lib.march_create.restype  = ctypes.c_void_p
        lib.march_create.argtypes = [ctypes.c_size_t, ctypes.c_uint32]

        lib.march_destroy.restype  = None
        lib.march_destroy.argtypes = [ctypes.c_void_p]

        lib.march_insert.restype  = ctypes.c_int
        lib.march_insert.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

        lib.march_query.restype  = ctypes.c_uint32
        lib.march_query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]

        lib.march_stats.restype  = None
        lib.march_stats.argtypes = [ctypes.c_void_p]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, tokens: list[int], kv_data: bytes | str) -> bool:
        """
        Insert a token sequence with its KV data.

        Returns True on success, False if the page pool is full.
        """
        arr = (ctypes.c_uint32 * len(tokens))(*tokens)
        if isinstance(kv_data, str):
            kv_data = kv_data.encode()
        buf = (ctypes.c_char * len(kv_data))(*kv_data)
        ok = self._lib.march_insert(self._ctx, arr, len(tokens), buf, len(kv_data))
        if ok:
            self._inserts += 1
        return bool(ok)

    def query(self, tokens: list[int]) -> tuple[int, int]:
        """
        Query the cache for prefix match.

        Returns
        -------
        (page_count, matched_tokens)
            page_count      — number of KV pages matched
            matched_tokens  — how many tokens of the prefix were found
        """
        n = len(tokens)
        arr = (ctypes.c_uint32 * n)(*tokens)
        out_ptrs = (ctypes.c_void_p * n)()
        matched = ctypes.c_uint32(0)
        count = self._lib.march_query(
            self._ctx, arr, n, out_ptrs, None, n, ctypes.byref(matched)
        )
        self._queries += 1
        self._total_queried += n
        self._total_matched += matched.value
        if matched.value > 0:
            self._hits += 1
        return int(count), int(matched.value)

    def stats(self) -> str:
        """Return a plain-text statistics summary."""
        elapsed = time.monotonic() - self._start_time
        hit_rate = (self._hits / self._queries * 100) if self._queries else 0.0
        token_reuse = (
            (self._total_matched / self._total_queried * 100)
            if self._total_queried else 0.0
        )
        lines = [
            "── March Stats ──────────────────────────",
            f"  uptime          : {elapsed:.1f}s",
            f"  inserts         : {self._inserts}",
            f"  queries         : {self._queries}",
            f"  hit rate        : {hit_rate:.1f}%  ({self._hits}/{self._queries})",
            f"  token reuse     : {token_reuse:.1f}%  "
            f"({self._total_matched}/{self._total_queried} tokens served from cache)",
            f"  page pool       : {self.max_pages} pages × {self.page_size} B"
            f" = {self.max_pages * self.page_size / 1024:.1f} KB total",
            "─────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def print_stats(self):
        print(self.stats())

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx and hasattr(self, "_lib"):
            self._lib.march_destroy(self._ctx)
            self._ctx = None
