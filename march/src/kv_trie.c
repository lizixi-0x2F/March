#include "kv_trie.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static uint64_t get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ============================================================
 * TrieMap — open-addressing HashMap, key=uint32_t token_id
 * ============================================================ */

static void map_init(TrieMap *m) {
    m->cap     = TRIE_MAP_INIT_CAP;
    m->size    = 0;
    m->buckets = calloc(m->cap, sizeof(TrieMapEntry));
}

static void map_free(TrieMap *m) {
    free(m->buckets);
    m->buckets = NULL;
    m->cap = m->size = 0;
}

/* internal slot lookup (linear probing) */
static uint32_t map_slot(const TrieMap *m, uint32_t key) {
    uint32_t idx = (key * 2654435761u) & (m->cap - 1);
    while (m->buckets[idx].occupied && m->buckets[idx].key != key)
        idx = (idx + 1) & (m->cap - 1);
    return idx;
}

static TrieNode *map_get(const TrieMap *m, uint32_t key) {
    uint32_t idx = map_slot(m, key);
    if (m->buckets[idx].occupied && m->buckets[idx].key == key)
        return m->buckets[idx].val;
    return NULL;
}

/* grow capacity and rehash */
static void map_grow(TrieMap *m) {
    uint32_t old_cap = m->cap;
    TrieMapEntry *old = m->buckets;

    m->cap    *= 2;
    m->size    = 0;
    m->buckets = calloc(m->cap, sizeof(TrieMapEntry));

    for (uint32_t i = 0; i < old_cap; i++) {
        if (!old[i].occupied) continue;
        uint32_t idx = map_slot(m, old[i].key);
        m->buckets[idx].key      = old[i].key;
        m->buckets[idx].val      = old[i].val;
        m->buckets[idx].occupied = 1;
        m->size++;
    }
    free(old);
}

static void map_put(TrieMap *m, uint32_t key, TrieNode *val) {
    if (m->size * 2 >= m->cap) map_grow(m);
    uint32_t idx = map_slot(m, key);
    m->buckets[idx].key      = key;
    m->buckets[idx].val      = val;
    m->buckets[idx].occupied = 1;
    m->size++;
}

/* ============================================================
 * TrieNode
 * ============================================================ */

static TrieNode *node_create(uint32_t token_id) {
    TrieNode *n   = calloc(1, sizeof(TrieNode));
    n->token_id   = token_id;
    n->page       = NULL;
    map_init(&n->children);
    return n;
}

static void node_destroy(TrieNode *n, PageAllocator *pa) {
    if (!n) return;
    /* recursively destroy all children */
    for (uint32_t i = 0; i < n->children.cap; i++) {
        if (n->children.buckets[i].occupied)
            node_destroy(n->children.buckets[i].val, pa);
    }
    map_free(&n->children);
    if (n->page) pa_free(pa, n->page);
    free(n);
}

/* ============================================================
 * KVTrie public API
 * ============================================================ */

KVTrie *trie_create(PageAllocator *pa) {
    KVTrie *t  = calloc(1, sizeof(KVTrie));
    t->pa      = pa;
    t->root    = node_create(0);  /* root node token_id is unused */
    t->node_count = 1;
    return t;
}

void trie_destroy(KVTrie *trie) {
    if (!trie) return;
    node_destroy(trie->root, trie->pa);
    free(trie);
}

KVPage *trie_insert(KVTrie *trie,
                    const uint32_t *token_ids, uint32_t n,
                    const void *kv_data, size_t kv_len)
{
    TrieNode *cur = trie->root;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t tid  = token_ids[i];
        TrieNode *child = map_get(&cur->children, tid);

        if (!child) {
            child = node_create(tid);
            map_put(&cur->children, tid, child);
            trie->node_count++;
        }
        cur = child;
    }

    /* leaf node: reuse existing page (shared prefix) or allocate a new one */
    if (!cur->page) {
        cur->page = pa_alloc(trie->pa);
        if (!cur->page) return NULL;   /* pool full */
    }

    /* write KV data (truncated to page size) */
    if (kv_data && kv_len > 0) {
        size_t copy_len = kv_len < trie->pa->page_size
                          ? kv_len : trie->pa->page_size;
        memcpy(cur->page->data, kv_data, copy_len);
    }

    return cur->page;
}

KVPage *trie_lookup(const KVTrie *trie,
                    const uint32_t *token_ids, uint32_t n,
                    uint32_t *prefix_len)
{
    TrieNode *cur  = trie->root;
    uint32_t  matched = 0;

    for (uint32_t i = 0; i < n; i++) {
        TrieNode *child = map_get(&cur->children, token_ids[i]);
        if (!child) break;
        cur = child;
        matched++;
    }

    if (prefix_len) *prefix_len = matched;

    /* only a full match with a page at the leaf counts as a hit */
    if (matched == n && cur->page)
        return cur->page;
    return NULL;
}

uint32_t trie_collect_path(const KVTrie *trie,
                           const uint32_t *token_ids, uint32_t n,
                           KVPage **out_pages, uint32_t capacity,
                           uint32_t *matched_tokens)
{
    TrieNode *cur   = trie->root;
    uint32_t  count = 0;
    uint64_t  now   = get_timestamp();
    if (matched_tokens) *matched_tokens = 0;

    for (uint32_t i = 0; i < n; i++) {
        TrieNode *child = map_get(&cur->children, token_ids[i]);
        if (!child) break;
        cur = child;
        if (matched_tokens) (*matched_tokens)++;
        if (cur->page && count < capacity) {
            cur->page->last_used = now;
            out_pages[count++] = cur->page;
        }
    }
    return count;
}

void trie_stats(const KVTrie *trie) {
    printf("[KVTrie] nodes=%llu\n", (unsigned long long)trie->node_count);
}
