import torch

import explicit_priors as ep


def test_learn_filters(natural):
    f = ep.learn_filters(natural, n_filters=4, size=3, sigma=0.1, steps=30, batch=4, crop=32, restarts=1)
    assert f.shape == (4, 3, 3) and torch.isfinite(f).all()
    assert torch.allclose(f.mean((1, 2)), torch.zeros(4, device=f.device), atol=1e-6)
    prior = ep.fit(f, natural, sigma=0.1)
    assert torch.isfinite(prior.coef).all()
