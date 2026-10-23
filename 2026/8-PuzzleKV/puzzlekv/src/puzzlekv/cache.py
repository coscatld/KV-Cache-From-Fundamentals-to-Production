from dataclasses import dataclass
import sys
import torch
from .decomposition import Factors, decompose

@dataclass
class Page:
    start: int
    length: int
    keys: list
    values: list
    @property
    def compressed(self):
        return isinstance(self.keys[0], Factors)

class KVCache:
    """One layer, batch=1; input [KV heads, tokens, dimension], post-RoPE keys."""
    def __init__(self, config):
        self.config, self.pages, self.length = config, [], 0
        self.layout = None

    def append(self, k, v):
        c = self.config
        if k.ndim != 3 or v.ndim != 3 or k.shape[:2] != v.shape[:2]:
            raise ValueError("expected matching [H,T,D] tensors")
        layout = (k.shape[0], k.shape[2], v.shape[2], k.dtype, k.device)
        expected_device = torch.device(c.device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if k.dtype != v.dtype or k.device != v.device or k.dtype != c.compute_dtype or k.device != expected_device:
            raise ValueError("dtype/device must match configuration")
        if self.layout is not None and layout != self.layout:
            raise ValueError("cache layout changed")
        self.layout = layout
        for t in range(k.shape[1]):
            if not self.pages or self.pages[-1].length == c.page_size:
                self.pages.append(Page(self.length, 0, [k[h,t:t+1].detach().clone() for h in range(k.shape[0])], [v[h,t:t+1].detach().clone() for h in range(v.shape[0])]))
            else:
                p = self.pages[-1]
                p.keys = [torch.cat((x, k[h,t:t+1].detach())) for h,x in enumerate(p.keys)]
                p.values = [torch.cat((x, v[h,t:t+1].detach())) for h,x in enumerate(p.values)]
            self.pages[-1].length += 1
            self.length += 1
        self.compress()

    def compress(self):
        c = self.config
        for p in self.pages:
            if c.compression_enabled and not p.compressed and p.length == c.page_size and p.start >= c.sink_size and p.start+p.length <= self.length-c.recent_window_size:
                # Preserve dense pages when factors cannot reduce tensor storage.
                if all(min(r, *x.shape)*(sum(x.shape)) < x.numel() for xs,r in ((p.keys,c.key_rank),(p.values,c.value_rank)) for x in xs):
                    p.keys = [decompose(x,c.key_rank) for x in p.keys]
                    p.values = [decompose(x,c.value_rank) for x in p.values]

    def storage(self):
        payload = sum(x.nbytes if isinstance(x,Factors) else x.numel()*x.element_size() for p in self.pages for x in p.keys+p.values)
        # Shallow Python metadata accounting, excludes tensor objects/allocator overhead.
        metadata = sys.getsizeof(self.pages)+sum(sys.getsizeof(p)+sys.getsizeof(p.__dict__)+sys.getsizeof(p.keys)+sys.getsizeof(p.values)+sys.getsizeof(p.start)+sys.getsizeof(p.length) for p in self.pages)
        h,d,dv,dtype,_ = self.layout
        return dict(tensor_bytes=payload, metadata_shallow_bytes=metadata, full_tensor_bytes=h*self.length*(d+dv)*torch.empty((),dtype=dtype).element_size())
