# March

*Trie-based KV cache with prefix sharing for LLM inference.*

---

When multiple LLM requests share a system prompt or conversation history, a naive cache stores N full copies of the same key/value tensors. With 100 concurrent sessions on a 512-token system prompt, that's 100 identical blobs in GPU memory. The prefix is identical. The storage isn't.

March fixes that. One copy, shared by everyone.

---

## How it works

March inserts token sequences into a Trie. Nodes that share a path share physical memory. A fixed-size mmap'd pool manages pages. Reads return raw pointers into that pool — zero memcpy, zero allocation.

```
insert([sys, A, B, C],    kv₁)
insert([sys, A, B, C, D], kv₂)

  root → [sys] → [A] → [B] → [C] ← KVPage  (shared by both)
                                └── [D] ← KVPage  (one new page)

query([sys, A, B, C, D])
  → 5 hops, each O(1) via internal hashmap
  → returns pointers: [&page_C, &page_D]
  → zero memcpy, zero allocation
```

Each node's children live in an open-addressing hashmap with Fibonacci hashing, so each hop is O(1) regardless of vocabulary size. The overall lookup is O(L) where L is sequence length — which is the minimum possible, since you must traverse the prefix.

**Memory:**
- Traditional: `N × L × page_size`
- March: `unique_nodes × page_size`
- Savings: 80–97% in prefix-heavy workloads

---

## Install

```bash
# 从 GitHub 一条命令安装（自动编译 C 核心）
uv pip install git+https://github.com/lizixi-0x2F/March

# 带 HuggingFace 集成
uv pip install "march[hf] @ git+https://github.com/lizixi-0x2F/March"

# 本地开发
git clone https://github.com/lizixi-0x2F/March && cd March
uv pip install -e ".[hf]"
```

需要：`gcc`，Python ≥ 3.10。HF 集成额外需要 `transformers ≥ 4.40` 和 `torch`。

---

## Usage

### Core cache

```python
from march import MarchCache

cache = MarchCache(page_size=4096, max_pages=256)

cache.insert([1, 2, 3], kv_bytes)
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

### HuggingFace integration

`MarchPrefixCache` wraps a HF model and intercepts prefill. On a cache hit it skips the forward pass entirely and returns stored `past_key_values` directly.

```python
from march.hf import MarchPrefixCache

prefix_cache = MarchPrefixCache(model, tokenizer)

# Miss — runs prefill, caches result
past_kv, hit = prefix_cache.prefill(input_ids)   # hit=False

# Same prefix again — no forward pass
past_kv, hit = prefix_cache.prefill(input_ids)   # hit=True

outputs = model.generate(
    input_ids,
    past_key_values=past_kv,
    use_cache=True,
)

prefix_cache.print_stats()
```

---

## When March wins

| Workload | Savings | Why |
|---|---|---|
| Multi-turn chat | 80–97% | Same system prompt every request |
| Batch inference | 80–97% | N requests, 1 shared prefix copy |
| Speculative / beam search | 40–70% | Branching from a shared root |
| Random sequences | ~0% | No shared prefixes to deduplicate |

---

## Architecture

Three C modules. One Python binding layer. No runtime dependencies.

**PageAllocator** — mmap-backed fixed-size pool. Alloc and free are O(1) via a free-list stack. Pages are reference-counted for safe sharing. LRU timestamps enable eviction when the pool fills.

**KVTrie** — Token-keyed prefix trie. Each node holds an open-addressing hashmap over its children, so vocabulary size doesn't affect lookup speed. Nodes without KV data act as internal routing nodes only.

**ViewBuilder** — Walks a path through the Trie and fills a caller-provided pointer array with references to live pool pages, root-to-leaf. The inference engine reads directly from the mmap'd buffer.

```
march_insert → KVTrie → PageAllocator  (alloc page, memcpy kv_data in)
march_query  → KVTrie → ViewBuilder    → out_ptrs[]  (zero-copy into pool)
```

---

## Demo

```bash
python3 march/demo/demo_basic.py
```

```bash
pip install transformers torch
python3 march/demo/demo_smollm2_memory.py
```

![SmolLM2 Memory Comparison](smollm2_memory_comparison.png)

*500 multi-turn conversations, same system prompt. Left: traditional. Right: March.*

---

March is early-stage research software. The C core is stable; the Python API may evolve.

MIT License
