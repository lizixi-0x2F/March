"""
march — Trie-based KV cache with prefix sharing for LLM inference.
"""
from march._core import MarchCache

__all__ = ["MarchCache"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "MarchPrefixCache":
        from march.hf import MarchPrefixCache
        return MarchPrefixCache
    raise AttributeError(f"module 'march' has no attribute {name!r}")
