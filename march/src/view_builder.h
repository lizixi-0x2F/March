#pragma once
#include <stdint.h>
#include "page_allocator.h"
#include "kv_trie.h"

/*
 * KVView — the "virtually contiguous" view returned by a single query.
 * pages[i] is a pointer to a physical KVPage, ordered per token segment.
 * The upper inference engine reads each page's data pointer in order,
 * requiring no memcpy at all.
 */
typedef struct KVView {
    KVPage  **pages;          /* pointer array, each entry points to a physical page */
    uint32_t  count;          /* number of pages */
    uint32_t  matched_tokens; /* actual number of matched tokens */
} KVView;

/*
 * Build a view: prefix-match token_ids[0..n) in the trie, collect all
 * page-bearing nodes along the root-to-match path, and return them as a KVView.
 *
 * token_ids[0..n) is the full token sequence of the current inference request.
 * The returned view must be released with view_free() (only the pointer array
 * is freed; physical pages are untouched).
 */
KVView *view_build(const KVTrie *trie,
                   const uint32_t *token_ids, uint32_t n);

/* Free the view struct (physical pages are untouched) */
void view_free(KVView *view);

/* Print view contents (for debugging) */
void view_print(const KVView *view);
