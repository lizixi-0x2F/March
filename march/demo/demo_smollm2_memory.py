#!/usr/bin/env python3
"""
SmolLM2 Memory Usage Comparison
Focus: Memory savings through prefix sharing in multi-conversation scenarios
"""
import ctypes
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

class MarchKVCache:
    def __init__(self, lib_path="march/libmarch.so", page_size=4096, max_pages=10000):
        self.lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()
        self.ctx = self.lib.march_create(page_size, max_pages)
        self.page_size = page_size
        self.max_pages = max_pages
        self.insert_count = 0

    def _setup_ctypes(self):
        self.lib.march_create.restype = ctypes.c_void_p
        self.lib.march_create.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
        self.lib.march_destroy.argtypes = [ctypes.c_void_p]
        self.lib.march_insert.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.march_insert.restype = ctypes.c_int

    def insert(self, token_ids):
        arr = (ctypes.c_uint32 * len(token_ids))(*token_ids)
        kv_data = ctypes.create_string_buffer(self.page_size)
        if self.lib.march_insert(self.ctx, arr, len(token_ids), kv_data, self.page_size):
            self.insert_count += 1
            return True
        return False

    def memory_usage_mb(self):
        return (self.max_pages * self.page_size) / (1024 * 1024)

    def __del__(self):
        if hasattr(self, 'ctx'):
            self.lib.march_destroy(self.ctx)

def test_memory_comparison():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

    system_prompt = "You are a helpful AI assistant."
    system_tokens = tokenizer.encode(system_prompt)

    num_conversations = 500
    turns_per_conv = 10

    print(f"\nSimulating {num_conversations} conversations with {turns_per_conv} turns each")
    print(f"System prompt: {len(system_tokens)} tokens (shared)")
    print("="*60)

    # Baseline
    baseline_cache = {}
    baseline_total_tokens = 0
    baseline_memory_trend = []

    for conv_id in range(num_conversations):
        history = system_tokens.copy()
        for turn in range(turns_per_conv):
            user_tokens = [1000 + conv_id * 10 + turn * 2] * 10
            history.extend(user_tokens)
            baseline_cache[tuple(history)] = b"kv"
            baseline_total_tokens += len(history)

            assistant_tokens = [2000 + conv_id * 10 + turn * 2 + 1] * 15
            history.extend(assistant_tokens)
            baseline_cache[tuple(history)] = b"kv"
            baseline_total_tokens += len(history)

            if conv_id == 0:
                baseline_memory_trend.append(baseline_total_tokens * 4096 / (1024 * 1024))

    baseline_memory_mb = (baseline_total_tokens * 4096) / (1024 * 1024)
    print(f"\n[Baseline] Dict Storage:")
    print(f"  Entries: {len(baseline_cache)}")
    print(f"  Total tokens: {baseline_total_tokens:,}")
    print(f"  Memory: {baseline_memory_mb:.1f} MB")

    # March
    march = MarchKVCache(page_size=4096, max_pages=50000)
    march_memory_trend = []

    for conv_id in range(num_conversations):
        history = system_tokens.copy()
        for turn in range(turns_per_conv):
            user_tokens = [1000 + conv_id * 10 + turn * 2] * 10
            history.extend(user_tokens)
            march.insert(history)

            assistant_tokens = [2000 + conv_id * 10 + turn * 2 + 1] * 15
            history.extend(assistant_tokens)
            march.insert(history)

            if conv_id == 0:
                march_memory_trend.append(march.insert_count * 4096 / (1024 * 1024))

    march_memory_mb = march.memory_usage_mb()
    print(f"\n[March] Trie Storage:")
    print(f"  Inserts: {march.insert_count}")
    print(f"  Memory: {march_memory_mb:.1f} MB")

    savings = (1 - march_memory_mb/baseline_memory_mb) * 100
    print(f"\n{'='*60}")
    print(f"Memory Savings: {savings:.1f}%")
    print(f"Reduction: {baseline_memory_mb:.1f} MB → {march_memory_mb:.1f} MB")

    # Plot
    sns.set_theme(style="whitegrid", palette="pastel")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: total memory
    methods = ['Baseline (Dict)', 'March (Trie)']
    memory = [baseline_memory_mb, march_memory_mb]
    bars = ax1.bar(methods, memory, color=sns.color_palette("pastel")[:2])

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB', ha='center', va='bottom')

    ax1.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax1.set_title(f'Total Memory: {savings:.1f}% Savings', fontsize=13, fontweight='bold')

    # Line chart: memory growth trend
    turns = list(range(1, len(baseline_memory_trend) + 1))
    ax2.plot(turns, baseline_memory_trend, marker='o', label='Baseline',
             color=sns.color_palette("pastel")[4], linewidth=2)
    ax2.plot(turns, march_memory_trend, marker='s', label='March',
             color=sns.color_palette("pastel")[0], linewidth=2)
    ax2.set_xlabel('Conversation Turns', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Growth Over Turns', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smollm2_memory_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to smollm2_memory_comparison.png")

if __name__ == "__main__":
    test_memory_comparison()

