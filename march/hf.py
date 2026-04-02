"""
march.hf — HuggingFace Transformers integration.

Provides MarchPrefixCache: a drop-in prefix-KV store that works alongside
transformers' standard past_key_values.  It does NOT try to replace
DynamicCache internally (the new CacheLayerMixin protocol makes that
impractical without forking transformers).  Instead it provides a simple
two-step workflow:

    cache = MarchPrefixCache(model, tokenizer)

    # During prefill (first call):
    result = cache.prefill(input_ids)
    #   → if prefix hit: returns pre-computed past_key_values, skips prefill
    #   → if miss:       runs model, stores result, returns past_key_values

    # After generation, save the new prefix:
    cache.save(input_ids, result.past_key_values)

This gives you:
  - Automatic prefix reuse across requests
  - Zero-copy reads for matched pages (stored as raw bytes in MarchCache)
  - Plain-text stats via cache.stats()
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from march._core import MarchCache

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# Sentinel so we can detect "not yet imported"
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch as _t
            _torch = _t
        except ImportError as e:
            raise ImportError("torch is required for march.hf") from e
    return _torch


def _kv_to_bytes(past_key_values) -> bytes:
    """
    Serialize a transformers past_key_values tuple-of-tuples into raw bytes.
    Layout: [n_layers uint32][for each layer: key_shape, key_bytes, val_shape, val_bytes]
    """
    torch = _get_torch()
    parts = []
    layers = (
        past_key_values.to_legacy_cache()
        if hasattr(past_key_values, "to_legacy_cache")
        else past_key_values
    )
    parts.append(struct.pack("<I", len(layers)))
    for k, v in layers:
        for t in (k, v):
            t_cpu = t.detach().cpu().contiguous()
            shape = list(t_cpu.shape)
            dtype_str = str(t_cpu.dtype).encode()  # e.g. b'torch.float32'
            raw = t_cpu.numpy().tobytes()
            # header: ndim, shape dims, dtype_len, dtype, data_len, data
            parts.append(struct.pack("<I", len(shape)))
            parts.append(struct.pack(f"<{len(shape)}I", *shape))
            parts.append(struct.pack("<I", len(dtype_str)))
            parts.append(dtype_str)
            parts.append(struct.pack("<I", len(raw)))
            parts.append(raw)
    return b"".join(parts)


def _bytes_to_kv(data: bytes, device):
    """Deserialize bytes back to a tuple-of-tuples of torch tensors."""
    torch = _get_torch()
    import numpy as np

    pos = 0

    def read(n):
        nonlocal pos
        chunk = data[pos: pos + n]
        pos += n
        return chunk

    def read_uint32():
        return struct.unpack("<I", read(4))[0]

    def read_tensor():
        ndim = read_uint32()
        shape = struct.unpack(f"<{ndim}I", read(4 * ndim))
        dtype_len = read_uint32()
        dtype_str = read(dtype_len).decode()  # e.g. 'torch.float32'
        data_len = read_uint32()
        raw = read(data_len)
        # Map dtype string to numpy dtype
        np_dtype_map = {
            "torch.float32": np.float32,
            "torch.float16": np.float16,
            "torch.bfloat16": np.float32,  # numpy has no bfloat16
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
        }
        np_dtype = np_dtype_map.get(dtype_str, np.float32)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        t = torch.from_numpy(arr.copy()).to(device)
        if dtype_str == "torch.bfloat16":
            t = t.to(torch.bfloat16)
        return t

    n_layers = read_uint32()
    layers = []
    for _ in range(n_layers):
        k = read_tensor()
        v = read_tensor()
        layers.append((k, v))
    return tuple(layers)


class MarchPrefixCache:
    """
    Prefix KV cache backed by March, compatible with HuggingFace generate().

    Parameters
    ----------
    model : PreTrainedModel
        The HF model (used for device and to run prefill on cache miss).
    tokenizer : PreTrainedTokenizerBase
        The tokenizer (used to convert token ids for display in stats).
    page_size : int
        Bytes per March page.  Should be large enough to hold serialized KV
        for one typical prefix.  Defaults to 4 MB.
    max_pages : int
        Maximum number of prefix entries to cache.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizerBase",
        page_size: int = 4 * 1024 * 1024,
        max_pages: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._march = MarchCache(page_size=page_size, max_pages=max_pages)
        # extra stats
        self._prefill_runs = 0
        self._cache_hits = 0

    def _token_ids(self, input_ids) -> list[int]:
        torch = _get_torch()
        if isinstance(input_ids, torch.Tensor):
            return input_ids.flatten().tolist()
        return list(input_ids)

    def lookup(self, input_ids) -> tuple | None:
        """
        Look up prefix in cache.
        Returns past_key_values tuple if found, else None.
        """
        tokens = self._token_ids(input_ids)
        page_count, matched = self._march.query(tokens)
        if matched == 0 or page_count == 0:
            return None
        # Retrieve raw bytes from the matched page (first/only page for this prefix)
        # We stored the full serialized KV in a single page per prefix.
        # page_count > 0 means matched — read back via a direct re-query with ctypes.
        raw = self._read_page_bytes(tokens[:matched])
        if raw is None:
            return None
        self._cache_hits += 1
        return _bytes_to_kv(raw, self.device)

    def _read_page_bytes(self, tokens: list[int]) -> bytes | None:
        """Re-query and read raw bytes from the matching leaf page."""
        import ctypes
        n = len(tokens)
        if n == 0:
            return None
        arr = (ctypes.c_uint32 * n)(*tokens)
        out_ptrs = (ctypes.c_void_p * n)()
        matched = ctypes.c_uint32(0)
        count = self._march._lib.march_query(
            self._march._ctx, arr, n, out_ptrs, None, n, ctypes.byref(matched)
        )
        if count == 0 or not out_ptrs[count - 1]:
            return None
        ptr = out_ptrs[count - 1]
        raw = ctypes.string_at(ptr, self._march.page_size)
        # strip trailing null bytes
        return raw.rstrip(b"\x00")

    def save(self, input_ids, past_key_values) -> bool:
        """
        Serialize and store past_key_values for the given input_ids prefix.
        Returns True on success.
        """
        tokens = self._token_ids(input_ids)
        raw = _kv_to_bytes(past_key_values)
        return self._march.insert(tokens, raw)

    def prefill(self, input_ids, **_generate_kwargs):
        """
        Attempt a cache-hit prefill.

        - On hit : returns the cached past_key_values directly.
        - On miss: runs model forward pass, caches result, returns past_key_values.

        Returns
        -------
        past_key_values : tuple
            Ready to pass as `past_key_values=` to model.generate().
        hit : bool
            True if served from cache.
        """
        torch = _get_torch()
        cached = self.lookup(input_ids)
        if cached is not None:
            return cached, True

        # Cache miss — run prefill
        self._prefill_runs += 1
        with torch.no_grad():
            out = self.model(input_ids, use_cache=True)
        pkv = out.past_key_values
        self.save(input_ids, pkv)
        return pkv, False

    def stats(self) -> str:
        total = self._cache_hits + self._prefill_runs
        hit_rate = (self._cache_hits / total * 100) if total else 0.0
        lines = [
            "── MarchPrefixCache Stats ───────────────",
            f"  prefill runs    : {self._prefill_runs}",
            f"  cache hits      : {self._cache_hits}",
            f"  hit rate        : {hit_rate:.1f}%",
            self._march.stats(),
        ]
        return "\n".join(lines)

    def print_stats(self):
        print(self.stats())
