#include "page_allocator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>

static uint64_t get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static KVPage *pa_evict_lru(PageAllocator *pa) {
    KVPage *victim = NULL;
    uint64_t oldest = UINT64_MAX;

    for (uint32_t i = 0; i < pa->total_pages; i++) {
        KVPage *p = &pa->pages[i];
        if (p->state == PAGE_ACTIVE && p->ref_count == 0 && p->last_used < oldest) {
            oldest = p->last_used;
            victim = p;
        }
    }

    if (victim) {
        victim->state = PAGE_FREE;
        pa->free_list[pa->free_top++] = victim->page_id;
        pa->free_count++;
    }

    return victim;
}

PageAllocator *pa_create(size_t page_size, uint32_t total_pages) {
    PageAllocator *pa = calloc(1, sizeof(PageAllocator));
    if (!pa) return NULL;

    pa->page_size    = page_size;
    pa->total_pages  = total_pages;
    pa->free_count   = total_pages;

    /* allocate a contiguous raw memory pool via mmap */
    pa->pool = mmap(NULL, page_size * total_pages,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pa->pool == MAP_FAILED) {
        free(pa);
        return NULL;
    }

    /* allocate page descriptor array */
    pa->pages = calloc(total_pages, sizeof(KVPage));
    if (!pa->pages) goto err;

    /* allocate free page ID stack */
    pa->free_list = malloc(total_pages * sizeof(uint32_t));
    if (!pa->free_list) goto err;

    /* init: each descriptor points to its memory block, push all pages onto the stack */
    for (uint32_t i = 0; i < total_pages; i++) {
        pa->pages[i].data      = (char *)pa->pool + i * page_size;
        pa->pages[i].page_id   = i;
        pa->pages[i].state     = PAGE_FREE;
        pa->pages[i].ref_count = 0;
        pa->free_list[i]       = i;
    }
    pa->free_top = total_pages;

    return pa;

err:
    munmap(pa->pool, page_size * total_pages);
    free(pa->pages);
    free(pa->free_list);
    free(pa);
    return NULL;
}

KVPage *pa_alloc(PageAllocator *pa) {
    if (pa->free_top == 0) {
        /* pool full, try LRU eviction */
        if (!pa_evict_lru(pa)) return NULL;
    }

    uint32_t id = pa->free_list[--pa->free_top];
    pa->free_count--;

    KVPage *page    = &pa->pages[id];
    page->state     = PAGE_ACTIVE;
    page->ref_count = 1;
    page->seq_hash  = 0;
    page->last_used = get_timestamp();

    return page;
}

void pa_ref(KVPage *page) {
    if (page) page->ref_count++;
}

void pa_free(PageAllocator *pa, KVPage *page) {
    if (!page) return;
    if (page->ref_count > 0) page->ref_count--;
    if (page->ref_count > 0) return;   /* still referenced, do not reclaim */

    page->state = PAGE_FREE;
    pa->free_list[pa->free_top++] = page->page_id;
    pa->free_count++;
}

void pa_stats(const PageAllocator *pa) {
    printf("[PageAllocator] page_size=%zu  total=%u  free=%u  used=%u\n",
           pa->page_size, pa->total_pages,
           pa->free_count, pa->total_pages - pa->free_count);
}

void pa_destroy(PageAllocator *pa) {
    if (!pa) return;
    munmap(pa->pool, pa->page_size * pa->total_pages);
    free(pa->pages);
    free(pa->free_list);
    free(pa);
}
