#!/usr/bin/env python3
"""
Basic usage demonstration of March KV Cache
Shows insertion, query, and prefix sharing
"""
import ctypes

class MarchCache:
    def __init__(self, lib_path="march/libmarch.so", page_size=256, max_pages=64):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()
        self.ctx = self.lib.march_create(page_size, max_pages)
        self.page_size = page_size

    def _setup_ctypes(self):
        self.lib.march_create.restype = ctypes.c_void_p
        self.lib.march_create.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
        self.lib.march_destroy.argtypes = [ctypes.c_void_p]
        self.lib.march_insert.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.march_insert.restype = ctypes.c_int
        self.lib.march_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                         ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                         ctypes.POINTER(ctypes.c_uint32)]
        self.lib.march_query.restype = ctypes.c_uint32

    def insert(self, tokens, data):
        arr = (ctypes.c_uint32 * len(tokens))(*tokens)
        buf = ctypes.create_string_buffer(data.encode() if isinstance(data, str) else data)
        return self.lib.march_insert(self.ctx, arr, len(tokens), buf, len(buf))

    def query(self, tokens):
        arr = (ctypes.c_uint32 * len(tokens))(*tokens)
        out_ptrs = (ctypes.c_void_p * len(tokens))()
        matched = ctypes.c_uint32(0)
        count = self.lib.march_query(self.ctx, arr, len(tokens), out_ptrs, None,
                                     len(tokens), ctypes.byref(matched))
        return count, matched.value

    def __del__(self):
        if hasattr(self, 'ctx'):
            self.lib.march_destroy(self.ctx)

def main():
    print("=" * 60)
    print("March KV Cache - Basic Usage Demo")
    print("=" * 60)

    cache = MarchCache()

    # Example 1: Simple insertion and query
    print("\n[Example 1] Simple insertion and query")
    tokens1 = [10, 20, 30]
    cache.insert(tokens1, "kv-data-1")
    count, matched = cache.query(tokens1)
    print(f"  Inserted: {tokens1}")
    print(f"  Query result: {count} pages, {matched} tokens matched")

    # Example 2: Prefix sharing
    print("\n[Example 2] Prefix sharing")
    tokens2 = [10, 20, 30, 40]  # Extends tokens1
    cache.insert(tokens2, "kv-data-2")
    count, matched = cache.query(tokens2)
    print(f"  Inserted: {tokens2}")
    print(f"  Query result: {count} pages, {matched} tokens matched")
    print(f"  → Prefix [10, 20, 30] is shared, only [40] is new")

    # Example 3: Partial match
    print("\n[Example 3] Partial match")
    tokens3 = [10, 20, 99]  # Partial match
    count, matched = cache.query(tokens3)
    print(f"  Query: {tokens3}")
    print(f"  Result: {count} pages, {matched} tokens matched")
    print(f"  → Only [10, 20] matched, [99] not found")

if __name__ == "__main__":
    main()


