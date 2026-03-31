# March: A Trie-Based KV Cache Sharing Library

*— In the style of Richard Feynman*

---

## The Honest Question First

Before I tell you what this thing does, let me ask you something simple.

Imagine you're running a language model. A thousand users send you requests. Nine hundred of them start with the exact same system prompt — *"You are a helpful assistant..."* — followed by some conversation history, followed by their actual question.

Now here's what I want to know: **how many times do you compute the KV cache for that system prompt?**

If your answer is "a thousand times," you have a problem. You're doing the same work over and over, and worse, you're *storing* the same data over and over. That's wasteful. Not cleverly wasteful — just plain wasteful.

March is the fix.

---

## What Is a KV Cache, Really?

When a transformer processes tokens, it computes keys and values for each token — that's the KV cache. When you run the *same prefix* twice, you get *identical* keys and values. Physics doesn't surprise you here. Same input, same output. Always.

So the question becomes: why store it twice?

---

## The Core Idea: A Trie

Here's where it gets beautiful.

A **Trie** is a tree where each path from root to leaf spells out a sequence. Like this:

```
                    [root]
                   /      \
               [tok:10]  [tok:99]
              /        \
         [tok:20]    [tok:21]
            |
         [tok:30]
            |
         [tok:40]
```

If three users send sequences that all start with `[10, 20, 30]`, that prefix lives **once** in the tree. One node. One KV page. Three users sharing it.

March builds exactly this tree — but instead of storing words, it stores **token IDs**, and at each node it attaches a **KV page**: the actual cached attention keys and values for that token position.

```
Token sequence:  [10] → [20] → [30] → [40]
                  │       │       │       │
KV pages:        P₁     P₂      P₃     P₄

Insert sequence: [10] → [20] → [99]
                  │       │       │
KV pages:        P₁     P₂      P₅   ← P₁ and P₂ already exist, reused!
```

The prefix `[10, 20]` is **shared**. Stored once. Referenced by both sequences.

---

## The Memory Math

Traditional approach:

```
N requests × L tokens × page_size bytes
= N × L × S   bytes total
```

March:

```
unique_prefix_nodes × page_size bytes
= far fewer nodes × S   bytes total
```

In practice, with multi-turn conversations where everyone shares a system prompt and conversation history:

```
Traditional:  ████████████████████████████████  100%
March:        ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    3%
              └──┘
           only unique parts stored
```

**80–97% memory reduction.** Not because we compressed anything. Because we stopped duplicating.

---

## The Architecture: Three Moving Parts

```
┌─────────────────────────────────────────────┐
│              Your Code (Caller)             │
│   march_insert(tokens, kv_data)             │
│   march_query(tokens) → pointers           │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                 KVTrie                      │
│  A tree of TrieNodes, one per token.        │
│  Each node holds a pointer to a KVPage.     │
│                                             │
│  [root] ─→ [tok_A] ─→ [tok_B] ─→ [tok_C]  │
│                │            └──→ [KVPage₃] │
│             [KVPage₂]                       │
└──────────────────────┬──────────────────────┘
                       │ pa_alloc / pa_free
                       ▼
┌─────────────────────────────────────────────┐
│              PageAllocator                  │
│                                             │
│  One big mmap() slab, cut into fixed pages. │
│  A free-list stack tracks available pages.  │
│                                             │
│  pool: [ P₀ | P₁ | P₂ | P₃ | ... | Pₙ ]  │
│          ↑               ↑                  │
│        ACTIVE           FREE                │
│                                             │
│  pa_alloc() → pop from free stack: O(1)    │
│  pa_free()  → push back, decrement refcount │
└──────────────────────┬──────────────────────┘
                       │ zero-copy pointers
                       ▼
┌─────────────────────────────────────────────┐
│              ViewBuilder                    │
│                                             │
│  On query, walks the trie from root,        │
│  collects matching KVPage pointers          │
│  into a KVView{pages[], count}.             │
│                                             │
│  Returns direct pointers into the mmap      │
│  pool. No memcpy. No allocation.            │
│  Just: here's where the data lives.         │
└─────────────────────────────────────────────┘
```

---

## Insert vs. Query: Two Directions Through the Same Tree

**Insert** walks *down*, creating nodes as needed:

```
march_insert([A, B, C], kv_data)

root
 └─ A  ← already exists? reuse it
     └─ B  ← already exists? reuse it
         └─ C  ← new! allocate a KVPage, copy kv_data in
```

**Query** walks *down*, collecting pages:

```
march_query([A, B, C])

root → A → B → C
       ↓   ↓   ↓
      P₁  P₂  P₃  ← KVView: [&P₁, &P₂, &P₃]

Inference engine reads P₁, P₂, P₃ directly.
Zero memcpy. These are raw pointers into mmap.
```

---

## The Trade-off, Stated Plainly

| | Hash Map | March (Trie) |
|---|---|---|
| Lookup | O(1) | O(L) where L = sequence length |
| Memory | N × L × S | unique_nodes × S |
| Prefix sharing | No | Yes |
| Best for | Random sequences | Shared-prefix workloads |

March is **not** faster at lookup. It's smarter about what it stores. In LLM inference — where the same system prompt appears in thousands of requests — that's the right trade-off.

---

## The LRU Safety Valve

Memory pools are finite. What happens when they fill up?

March evicts the **least recently used** page — the one that hasn't been accessed in the longest time. Each `KVPage` carries a `last_used` timestamp (a logical clock). When `pa_alloc()` finds no free pages, it scans for the stalest one, evicts it, and reassigns.

```
Free list empty?
       │
       ▼
Find min(last_used) across ACTIVE pages
       │
       ▼
Detach from trie node → page.state = FREE
       │
       ▼
Return page to new requester
```

Old data makes room for new data. The pool never grows beyond its fixed size.

---

## Summary

March is a **prefix-sharing KV cache library** for LLM inference.

It works because:
1. Identical token prefixes produce identical KV data
2. A Trie naturally deduplicates shared prefixes
3. A fixed memory pool with reference counting makes sharing safe
4. Zero-copy queries return raw pointers — no extra allocation on the hot path

The result: in multi-turn conversation or batch inference with shared prompts, **80–97% less memory** than storing each sequence independently.

Same math. Fewer copies. That's it.

---

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."*
*— R. P. Feynman*

*Don't fool yourself into thinking duplicating data is free. It isn't.*
