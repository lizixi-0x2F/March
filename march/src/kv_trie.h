#pragma once
#include <stdint.h>
#include "page_allocator.h"

/* ---- Internal HashMap (open addressing, linear probing) ---- */
#define TRIE_MAP_INIT_CAP 8

typedef struct TrieMapEntry {
    uint32_t   key;      /* token_id; 0 means empty slot (token 0 handled specially) */
    int        occupied; /* whether the slot is occupied */
    struct TrieNode *val;
} TrieMapEntry;

typedef struct TrieMap {
    TrieMapEntry *buckets;
    uint32_t      cap;
    uint32_t      size;
} TrieMap;

/* ---- Trie node ---- */
typedef struct TrieNode {
    uint32_t   token_id;   /* token represented by this node */
    KVPage    *page;       /* physical page for this prefix sequence (NULL for internal nodes) */
    TrieMap    children;   /* child node HashMap */
} TrieNode;

/* ---- Prefix Trie ---- */
typedef struct KVTrie {
    TrieNode     *root;
    PageAllocator *pa;    /* borrowed, not owned */
    uint64_t      node_count;
} KVTrie;

/* Create / Destroy */
KVTrie *trie_create(PageAllocator *pa);
void    trie_destroy(KVTrie *trie);

/*
 * Insert: write KV data for token_ids[0..n) into the Trie.
 * kv_data and kv_len are raw KV bytes provided by the caller; memcpy'd into a physical page.
 * Returns the leaf node's KVPage*, or NULL on failure.
 */
KVPage *trie_insert(KVTrie *trie,
                    const uint32_t *token_ids, uint32_t n,
                    const void *kv_data, size_t kv_len);

/*
 * Lookup: return the leaf KVPage* for an exact match of token_ids[0..n).
 * If only a prefix matches, prefix_len is set to the actual match length.
 * prefix_len may be NULL.
 */
KVPage *trie_lookup(const KVTrie *trie,
                    const uint32_t *token_ids, uint32_t n,
                    uint32_t *prefix_len);

/*
 * Path collection: prefix-match token_ids[0..n), writing the KVPage* of every
 * page-bearing node along the path into out_pages[0..capacity).
 * Returns actual count written; matched_tokens is set to actual matched token count.
 * out_pages is caller-allocated (recommended size >= n).
 */
uint32_t trie_collect_path(const KVTrie *trie,
                           const uint32_t *token_ids, uint32_t n,
                           KVPage **out_pages, uint32_t capacity,
                           uint32_t *matched_tokens);

/* Print trie statistics */
void trie_stats(const KVTrie *trie);
