import pytest
import torch

@pytest.fixture(autouse=True)
def cuda_required():
    if not torch.cuda.is_available():
        pytest.skip("NOT RUN: CUDA GPU required; CPU execution is not a validation target")
    torch.manual_seed(42)

@pytest.fixture
def device():
    return "cuda"
