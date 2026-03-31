#include "march.h"
#include "page_allocator.h"
#include "kv_trie.h"
#include "view_builder.h"
#include <stdlib.h>
#include <stdio.h>

struct MarchCtx {
    PageAllocator *pa;
    KVTrie        *trie;
};

MarchCtx *march_create(size_t page_size, uint32_t max_pages) {
    MarchCtx *ctx = malloc(sizeof(MarchCtx));
    if (!ctx) return NULL;

    ctx->pa = pa_create(page_size, max_pages);
    if (!ctx->pa) { free(ctx); return NULL; }

    ctx->trie = trie_create(ctx->pa);
    if (!ctx->trie) { pa_destroy(ctx->pa); free(ctx); return NULL; }

    return ctx;
}

void march_destroy(MarchCtx *ctx) {
    if (!ctx) return;
    trie_destroy(ctx->trie);   /* trie internally pa_free's each page */
    pa_destroy(ctx->pa);
    free(ctx);
}

int march_insert(MarchCtx *ctx,
                 const uint32_t *token_ids, uint32_t n,
                 const void *kv_data, size_t kv_len)
{
    KVPage *p = trie_insert(ctx->trie, token_ids, n, kv_data, kv_len);
    return p != NULL ? 1 : 0;
}

uint32_t march_query(MarchCtx *ctx,
                     const uint32_t *token_ids, uint32_t n,
                     void    **out_ptrs,
                     uint32_t *out_page_ids,
                     uint32_t  capacity,
                     uint32_t *matched_tokens)
{
    /* use view_builder to collect pages along the path */
    KVView *view = view_build(ctx->trie, token_ids, n);
    if (!view) return 0;

    uint32_t fill = view->count < capacity ? view->count : capacity;
    for (uint32_t i = 0; i < fill; i++) {
        if (out_ptrs)     out_ptrs[i]     = view->pages[i]->data;
        if (out_page_ids) out_page_ids[i] = view->pages[i]->page_id;
    }
    if (matched_tokens) *matched_tokens = view->matched_tokens;

    view_free(view);
    return fill;
}

void march_stats(MarchCtx *ctx) {
    pa_stats(ctx->pa);
    trie_stats(ctx->trie);
}
