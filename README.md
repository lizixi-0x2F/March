# March - High-Performance KV Cache Sharing Library

March is a **memory-efficient** KV Cache management system based on Trie structure, designed for LLM inference scenarios. It significantly **reduces memory usage** through prefix sharing, not by being faster, but by being smarter about storage.

## Why March?

**The Problem:** In LLM inference, multiple requests often share common prefixes (system prompts, conversation history). Traditional approaches store each sequence separately, wasting memory.

**The Solution:** March uses a Trie structure to automatically share identical prefixes across sequences, storing each prefix only once.

## Core Value Proposition

- **Memory Savings**: 80-97% reduction in scenarios with prefix overlap
- **Zero-Copy Access**: Query returns direct memory pointers without data copying
- **Predictable Memory**: Fixed-size page pool with O(L) complexity
- **Trade-off**: Slightly slower than dict O(1) lookup, but memory savings justify it

## Core Features

- **Prefix Sharing**: Automatically merges identical prefixes using Trie tree, storing each prefix only once
- **Zero-Copy Query**: Returns memory pointers directly without data copying
- **O(L) Complexity**: Both insertion and query are O(L) where L is sequence length
- **Memory Pool Management**: Fixed-size page allocation with predictable memory usage
- **C Implementation**: High-performance core with Python-friendly ctypes interface

**Key Insight**: March trades slightly slower query speed (O(L) vs O(1)) for massive memory savings (80-97%) in prefix-heavy workloads.

## Architecture

```
┌─────────────────────────────────────┐
│         Python / ctypes             │
├─────────────────────────────────────┤
│         march.h (API)               │
├─────────────────────────────────────┤
│  kv_trie.c  │  page_allocator.c     │
│  (Trie)   │  (Memory Page Pool)     │
└─────────────────────────────────────┘
```

### Core Components

- **PageAllocator**: Fixed-size page allocator managing physical memory pool
- **KVTrie**: Trie tree for token sequences, each node associated with a KV page
- **ViewBuilder**: Builds zero-copy views during queries, returns page pointer arrays

## Quick Start

### Build

```bash
cd march
make lib
```

### Basic Usage

```python
import ctypes
import os

# Load library
lib = ctypes.CDLL("march/libmarch.so")
lib.march_create.restype = ctypes.c_void_p
lib.march_create.argtypes = [ctypes.c_size_t, ctypes.c_uint32]

# Create context: 256 bytes/page, max 64 pages
ctx = lib.march_create(256, 64)

# Insert sequence
tokens = [10, 20, 30]
arr = (ctypes.c_uint32 * len(tokens))(*tokens)
data = ctypes.create_string_buffer(b"kv-cache-data")
lib.march_insert(ctx, arr, len(tokens), data, len(b"kv-cache-data"))

# Query
out_ptrs = (ctypes.c_void_p * len(tokens))()
matched = ctypes.c_uint32(0)
count = lib.march_query(ctx, arr, len(tokens), out_ptrs, None,
                        len(tokens), ctypes.byref(matched))

# Zero-copy read
for i in range(count):
    raw = (ctypes.c_char * 64).from_address(out_ptrs[i]).raw
    print(f"Page {i}: {raw}")

lib.march_destroy(ctx)
```

## Demo Examples

### 1. Basic Functionality
```bash
python3 march/demo/demo_basic.py
```
Demonstrates basic usage of insertion, query, and prefix sharing.

### 2. Performance Benchmark
```bash
python3 march/demo/demo_bench.py
```
Tests insertion and query throughput, validates O(L) time complexity.

### 3. HuggingFace Model Integration 🔥
```bash
pip install transformers torch
python3 march/demo/demo_hf_gpt.py
```
Loads GPT-2 model, uses March to manage KV Cache, compares memory usage and inference performance.

### 4. LLM Inference Benchmark 🔥
```bash
python3 march/demo/demo_llm_bench.py
```
Hardcore testing of three scenarios: batch inference, multi-turn dialogue, speculative sampling, showcasing March's memory and performance advantages.

### 6. Real LLM Integration Test (SmolLM2-135M) 🔥
```bash
pip install transformers torch
python3 march/demo/demo_smollm2_memory.py
```
Memory efficiency demonstration using HuggingFace's SmolLM2-135M model. Simulates 500 multi-turn conversations sharing a common system prompt, showing how March reduces memory footprint through prefix sharing.

![SmolLM2 Memory Comparison](smollm2_memory_comparison.png)

## API Reference

### march_create
```c
MarchCtx *march_create(size_t page_size, uint32_t max_pages);
```
Creates context with specified page size and maximum page count.

### march_insert
```c
int march_insert(MarchCtx *ctx, const uint32_t *token_ids, uint32_t n,
                 const void *kv_data, size_t kv_len);
```
Inserts token sequence and its KV data. Returns 1 on success, 0 on failure (memory pool full).

### march_query
```c
uint32_t march_query(MarchCtx *ctx, const uint32_t *token_ids, uint32_t n,
                     void **out_ptrs, uint32_t *out_page_ids,
                     uint32_t capacity, uint32_t *matched_tokens);
```
Queries sequence, returns number of matched pages. `out_ptrs` is zero-copy pointer array.

### march_destroy
```c
void march_destroy(MarchCtx *ctx);
```
Destroys context and frees all memory.

## Performance Metrics

Test results on M3 Mac:

| Metric | Value | Notes |
|--------|-------|-------|
| Memory Savings | 80-97% | In prefix-sharing scenarios |
| Insert Throughput | ~50,000 ops/s | Comparable to baseline |
| Query Throughput | ~200,000 ops/s | O(L) complexity |

**Memory comparison:**
- Traditional: N × L × page_size (each sequence stored separately)
- March: shared_nodes × page_size (prefixes stored once)

**When March wins:** Multi-turn conversations, batch inference with shared prompts, speculative sampling
**When dict wins:** Random sequences with no prefix overlap

## Use Cases

- **LLM Inference Service**: Reuse historical KV Cache in multi-turn conversations
- **Batch Inference**: Batch requests sharing prompt prefixes
- **Speculative Sampling**: Multiple candidate sequences sharing prefix KV
- **Tree Search**: KV management in Beam Search

## License

MIT License
