# March

> Trie-based KV cache with prefix sharing for LLM inference.

![SmolLM2 Memory Comparison](smollm2_memory_comparison.png)

*500 multi-turn conversations, same system prompt. Left: traditional storage. Right: March. The difference is prefix sharing — nothing else.*

When multiple requests share a common prefix (system prompt, few-shot examples, conversation history), a naive cache stores N full copies. March stores one copy and shares it — the Trie deduplicates at the token level, a fixed-size page pool manages physical memory, and all reads are zero-copy pointers into an mmap'd buffer.

---

## Install

```bash
cd march && make all   # build libmarch.so
pip install -e .       # or: uv pip install -e .

# with HuggingFace support
pip install -e ".[hf]"
```

## Quick Start

```python
from march import MarchCache

cache = MarchCache(page_size=4096, max_pages=256)

# Insert a token sequence with its KV bytes
cache.insert([1, 2, 3], kv_bytes)

# Query — returns (page_count, matched_tokens)
pages, matched = cache.query([1, 2, 3, 4])

cache.print_stats()
# ── March Stats ──────────────────────────
#   uptime          : 0.3s
#   inserts         : 1
#   queries         : 1
#   hit rate        : 100.0%  (1/1)
#   token reuse     : 75.0%   (3/4 tokens served from cache)
#   page pool       : 256 pages × 4096 B = 1024.0 KB total
# ─────────────────────────────────────────
```

## HuggingFace Integration

`MarchPrefixCache` wraps a HF model and intercepts prefill. On a cache hit it skips the forward pass entirely and returns the stored `past_key_values` directly.

```python
from march.hf import MarchPrefixCache

prefix_cache = MarchPrefixCache(model, tokenizer)

# First call — cache miss, runs prefill and stores result
past_kv, hit = prefix_cache.prefill(input_ids)   # hit=False

# Same prefix again — served from cache, no forward pass
past_kv, hit = prefix_cache.prefill(input_ids)   # hit=True

# Pass into generate as usual
outputs = model.generate(input_ids, past_key_values=past_kv, use_cache=True)

prefix_cache.print_stats()
```

## How It Works

```
insert([t0, t1, t2], kv_bytes)

  root → [t0] → [t1] → [t2] ← KVPage (mmap'd, ref-counted)
                              ↑
insert([t0, t1, t2, t3], ...) reuses this path, only allocates [t3]

query([t0, t1, t2, t3])
  → walks trie, each hop is O(1) via internal open-addressing hashmap
  → returns zero-copy pointers to pages[t0..t3]
  → overall O(L), L = sequence length
```

**Memory layout:**
- Traditional: `N × L × page_size` (each sequence stored separately)
- March: `shared_nodes × page_size` (prefixes stored once)
- Savings: 80–97% in prefix-heavy workloads

## When to Use

| Workload | Benefit |
|---|---|
| Multi-turn chat (shared system prompt) | High — same prefix every request |
| Batch inference (shared few-shot prefix) | High — N requests, 1 copy |
| Speculative sampling / beam search | Medium — branching prefixes |
| Fully random sequences | None |

## Architecture

Three C components, one Python binding layer:

- **PageAllocator** — mmap'd fixed-size pool, free-list stack, O(1) alloc/free
- **KVTrie** — token-keyed prefix trie, per-node open-addressing hashmap
- **ViewBuilder** — zero-copy path collection, root-to-leaf pointer arrays

```
march_insert → KVTrie → PageAllocator (alloc page, memcpy kv_data)
march_query  → KVTrie → ViewBuilder   → out_ptrs[] (zero-copy into pool)
```

## License

MIT
