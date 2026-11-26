"""Profiler: wall time and peak GPU memory, by phase data   : waiting for the DataLoader (decoding, augmentation) select : exploration without gradient, plus cutting the retained patches step   : forward and backward on the retained patches (or on the image for the baseline) The GPU is synchronised at each phase boundary, otherwise the time of an asynchronous kernel would be charged to the next phase. The peak counter is reset at the start of a phase, so each phase has its own peak; what is already allocated at that moment (weights, optimiser state, images that are on the GPU) is included in the peak, as it should be."""

import contextlib
import time
import torch
from collections import defaultdict

from ips_suite.config.environment import CUDA


class Profiler:
    def __init__(self):
        self.t = defaultdict(float)
        self.peak = defaultdict(float)
        self.n_img = 0

    @contextlib.contextmanager
    def phase(self, name):
        if CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if CUDA:
                torch.cuda.synchronize()
                self.peak[name] = max(self.peak[name],
                                      torch.cuda.max_memory_allocated())
            self.t[name] += time.perf_counter() - t0

    def summary(self, prefix=''):
        out = {f'{prefix}t_{k}_s': round(v, 3) for k, v in self.t.items()}
        out.update({f'{prefix}peak_{k}_mb': round(v / 1e6, 1)
                    for k, v in self.peak.items()})
        out[f'{prefix}peak_mb'] = round(max(self.peak.values(), default=0.0) / 1e6, 1)
        out[f'{prefix}n_img'] = self.n_img
        return out
