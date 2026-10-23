import time
from contextlib import contextmanager
import torch

class Metrics:
    def __init__(self,device="cpu"):
        self.device=device; self.seconds={}; self.events=[]; self.peak_kv_bytes=0
    def sync(self):
        if str(self.device).startswith("cuda"): torch.cuda.synchronize(self.device)
    @contextmanager
    def timer(self,name):
        self.sync(); start=time.perf_counter()
        try: yield
        finally:
            self.sync(); self.seconds[name]=self.seconds.get(name,0)+time.perf_counter()-start
    def observe(self,n): self.peak_kv_bytes=max(n,self.peak_kv_bytes)
