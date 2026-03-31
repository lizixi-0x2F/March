#include "view_builder.h"
#include <stdlib.h>
#include <stdio.h>

KVView *view_build(const KVTrie *trie,
                   const uint32_t *token_ids, uint32_t n)
{
    KVView  *view = malloc(sizeof(KVView));
    view->pages   = malloc(n * sizeof(KVPage *));
    view->count   = 0;
    view->matched_tokens = 0;

    view->count = trie_collect_path(trie, token_ids, n,
                                    view->pages, n,
                                    &view->matched_tokens);
    return view;
}

void view_free(KVView *view) {
    if (!view) return;
    free(view->pages);   /* only free pointer array; physical pages untouched */
    free(view);
}

void view_print(const KVView *view) {
    if (!view) { printf("[KVView] NULL\n"); return; }
    printf("[KVView] matched_tokens=%u  pages=%u\n",
           view->matched_tokens, view->count);
    for (uint32_t i = 0; i < view->count; i++) {
        KVPage *p = view->pages[i];
        printf("  [%u] page_id=%u  data=%p  ref=%u\n",
               i, p->page_id, p->data, p->ref_count);
    }
}
