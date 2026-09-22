import os

import torch


# Measured on a 16-core CPU with this model (~4.5k params) and batch_size=128:
# 1 thread = 4.9 s/epoch, 8 threads = 22.5 s, 16 threads = 331 s.
# The net is far too small to parallelise across many cores, so the OpenMP
# barrier cost dominates and more threads make training dramatically slower.
# A low cap is a free win on big machines and harmless on a laptop.
DEFAULT_CPU_THREADS = 4


def configure_threads(device):
    """Cap intra-op CPU threads. Override with TORCH_NUM_THREADS."""
    if device.type != "cpu":
        return torch.get_num_threads()

    requested = os.environ.get("TORCH_NUM_THREADS")

    if requested is not None:
        n_threads = int(requested)
    else:
        n_threads = min(DEFAULT_CPU_THREADS, os.cpu_count() or 1)

    torch.set_num_threads(n_threads)

    return n_threads
