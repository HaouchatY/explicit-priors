import torch

import explicit_priors as ep


def test_pretrained(device):
    assert "map_0.1" in ep.available()
    prior = ep.pretrained("map_0.1", map_location=device)
    assert prior.objective == "map" and abs(prior.sigma - 0.1) < 1e-12
    y = torch.rand(48, 48, device=device)
    x = ep.denoise(prior, y, iters=5)
    assert x.shape == y.shape and torch.isfinite(x).all()
