#pragma once
#include <stdint.h>
#include <stddef.h>

/*
 * march.h — unified public API
 *
 * Callers (Python ctypes or other C/C++ frameworks) only need this header.
 * All struct pointers are opaque externally; operate through functions.
 */

typedef struct MarchCtx MarchCtx;

/* ----- Lifecycle ----- */

/* Create context: page_size bytes/page, max_pages pages */
MarchCtx *march_create(size_t page_size, uint32_t max_pages);

/* Destroy and free all memory */
void march_destroy(MarchCtx *ctx);

/* ----- KV Write ----- */

/*
 * Write KV data for token_ids[0..n) into the cache.
 * kv_data/kv_len are raw bytes copied into a physical page (truncated to page_size).
 * Returns 1 on success, 0 if memory pool is full.
 */
int march_insert(MarchCtx *ctx,
                 const uint32_t *token_ids, uint32_t n,
                 const void *kv_data, size_t kv_len);

/* ----- KV Read (view) ----- */

/*
 * Query physical pages matching prefix token_ids[0..n), filling caller-provided arrays:
 *   out_ptrs[i]     — pointer to KV data of page i (direct pointer into pool, zero-copy)
 *   out_page_ids[i] — page index of page i
 *
 * Returns actual number of pages filled (<= capacity).
 * matched_tokens is set to the actual number of matched tokens (may be NULL).
 */
uint32_t march_query(MarchCtx *ctx,
                     const uint32_t *token_ids, uint32_t n,
                     void    **out_ptrs,
                     uint32_t *out_page_ids,
                     uint32_t  capacity,
                     uint32_t *matched_tokens);

/* ----- Stats ----- */
void march_stats(MarchCtx *ctx);
