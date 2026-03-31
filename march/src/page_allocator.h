#pragma once
#include <stddef.h>
#include <stdint.h>

/* Physical state of each page */
typedef enum {
    PAGE_FREE    = 0,
    PAGE_ACTIVE  = 1,
} PageState;

/* Physical KV page: holds KV tensor data for a fixed number of tokens */
typedef struct KVPage {
    void       *data;       /* pointer to actual KV data (within the memory pool) */
    uint64_t    seq_hash;   /* hash of the token sequence for this page, for fast matching */
    uint32_t    ref_count;  /* reference count: multiple Trie branches may share a page */
    uint32_t    page_id;    /* page index within the memory pool */
    uint64_t    last_used;  /* timestamp for LRU eviction (not yet enabled) */
    PageState   state;
} KVPage;

/* Page allocator */
typedef struct PageAllocator {
    void     *pool;         /* contiguous memory pool obtained via mmap */
    size_t    page_size;    /* bytes per page */
    uint32_t  total_pages;  /* total number of pages */
    uint32_t  free_count;   /* current number of free pages */
    KVPage   *pages;        /* array of page descriptors */
    uint32_t *free_list;    /* stack of free page IDs */
    uint32_t  free_top;     /* top-of-stack pointer for free_list */
} PageAllocator;

/* Initialize allocator: reserve total_pages pages of page_size bytes each */
PageAllocator *pa_create(size_t page_size, uint32_t total_pages);

/* Allocate one free page; returns NULL when pool is full */
KVPage *pa_alloc(PageAllocator *pa);

/* Release a page (actually reclaimed only when ref_count drops to zero) */
void pa_free(PageAllocator *pa, KVPage *page);

/* Increment reference count */
void pa_ref(KVPage *page);

/* Print allocator statistics */
void pa_stats(const PageAllocator *pa);

/* Destroy allocator and unmap memory */
void pa_destroy(PageAllocator *pa);
