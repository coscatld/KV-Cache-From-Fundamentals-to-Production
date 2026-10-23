from dataclasses import dataclass
import torch

@dataclass
class Config:
    page_size: int = 32
    key_rank: int = 8
    value_rank: int = 8
    sink_size: int = 32
    recent_window_size: int = 32
    compression_enabled: bool = True
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    numerical_tolerance: float = 1e-5

    def __post_init__(self):
        if min(self.page_size, self.key_rank, self.value_rank) < 1:
            raise ValueError("page_size and ranks must be positive")
        if min(self.sink_size, self.recent_window_size) < 0:
            raise ValueError("windows must be nonnegative")
        if self.compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported dtype")
        if self.numerical_tolerance <= 0:
            raise ValueError("tolerance must be positive")
