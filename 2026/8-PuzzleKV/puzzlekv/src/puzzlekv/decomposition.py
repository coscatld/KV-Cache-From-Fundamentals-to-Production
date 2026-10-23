from dataclasses import dataclass
import torch

@dataclass
class Factors:
    a: torch.Tensor
    b: torch.Tensor
    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in (self.a, self.b))
    def reconstruct(self):
        return self.a @ self.b

def rank_for_budget(rows, columns, budget_bytes, dtype=torch.float32):
    if min(rows, columns) < 1 or budget_bytes < 0:
        raise ValueError("invalid shape or budget")
    return min(rows, columns, budget_bytes // ((rows+columns)*torch.empty((), dtype=dtype).element_size()))

def decompose(x, rank):
    if x.ndim != 2 or min(x.shape) == 0 or rank < 1:
        raise ValueError("expected nonempty matrix and positive rank")
    rank = min(rank, *x.shape)
    u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return Factors((u[:, :rank]*s[:rank]).to(x.dtype).contiguous(), vh[:rank].to(x.dtype).contiguous())
