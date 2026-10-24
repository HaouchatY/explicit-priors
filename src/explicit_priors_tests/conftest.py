import pytest
import torch


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def natural(device):
    """32 smooth random 48 x 48 images in [0, 1] (low-passed noise)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 1, 64, 64, generator=g)
    k = torch.exp(-0.5 * (torch.arange(-6, 7) / 2.0) ** 2)
    k = torch.outer(k, k) / k.sum() ** 2
    x = torch.nn.functional.conv2d(x, k[None, None])[:, 0, :48, :48]
    x = (x - x.amin((1, 2), keepdim=True)) / (x.amax((1, 2), keepdim=True) - x.amin((1, 2), keepdim=True))
    return x.to(device)
