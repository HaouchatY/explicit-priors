# Explicit priors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fields-of-experts image priors whose potentials come from one linear solve.

<p align="center"><img src="doc/potentials.png" width="760" alt="DCT 3x3 filters and their MAP potentials"></p>

*Top: the 8 filters of the 3×3 DCT. Bottom: their potentials $\phi_k$, from
`ep.fit(ep.filters.dct(3), train, sigma=25 / 255, objective="map")` (monotone, the default for `"map"`: even and
non-decreasing in $|z|$, not convex) on the 400 BSD training images
(180×180, as in DnCNN and WCRR), 2 s on one H200. Each curve is shifted to a minimum of 0 and drawn up
to 1.6 times the response where it reaches 95% of its plateau. To reproduce:*
`python doc/make_figures.py --data path/to/BSD` *(with `train/` in `path/to/BSD`).*

A fields of experts scores an image through its filter responses,

$$\Phi(x) = \sum_k \sum_i \phi_k\big((f_k \star x)_i\big), \qquad \phi_k' = \sum_a c_{ka}\, b_a .$$

Its gradient is linear in the coefficients, $\nabla\Phi(x) = G(x)\,c$. Fitting a denoiser or a prior to
images is therefore a least-squares problem in $c$, solved by its normal equations $A c = d$: no
training loop, seconds on a GPU.

## Features

- **Potentials in closed form** for any filter bank, `ep.fit(filters, images, sigma, objective=...)`:
  - `"mse"`: the one-step denoiser $\hat x = y - \sigma^2 \nabla\Phi(y)$ with the lowest error.
  - `"map"`: the potentials whose MAP denoiser $\arg\min_x \tfrac12\|x-y\|^2 + \sigma^2\Phi(x)$ returns
    the MMSE estimate. `refine=n` adds Gauss–Newton steps on the MAP output itself.
  - `"score"`: the prior of clean images, by score matching.
- **Shapes**: `shape="free" | "monotone" | "convex"`.
- **Filters**: analytic (`dct`, `gabor`, `gaussian_derivatives`, `laplacian_of_gaussian`), from
  patches in closed form (`pca`, `ica`), learned with the potentials (`learn_filters`), or any
  (K, s, s) tensor. `symmetrize` ties the rotations of a filter to one potential.
- **Denoisers and a solver**: `one_step`, `map_estimate`, `denoise`, and `fista` for inverse
  problems.

PyTorch, on GPU or CPU.

## Installation

```bash
pip install "explicit_priors@git+https://github.com/HaouchatY/explicit-priors.git"

# developer install
git clone https://github.com/HaouchatY/explicit-priors.git
cd explicit-priors/
pip install -e ".[dev,examples]"
pre-commit install
```

## Getting started

```python
import explicit_priors as ep

# train: clean images in [0, 1], a (N, H, W) tensor or a list of (H, W) tensors
prior = ep.fit(ep.filters.dct(7), train, sigma=25 / 255, objective="map")
x = ep.denoise(prior, y)  # MAP with weight sigma^2: nothing to tune
prior.save("prior.pt")
```

Filters learned with the potentials:

```python
filters = ep.learn_filters(train, n_filters=16, size=7, sigma=25 / 255)
prior = ep.fit(filters, train, sigma=25 / 255)
```

Any inverse problem with a smooth data term, e.g. $\tfrac12\|Hx - y\|^2 + \lambda\Phi(x)$:

```python
def objective(x):
    r = H(x) - y
    e, g = prior.energy_grad(x)
    return 0.5 * (r * r).sum((1, 2)) + lam * e, Ht(r) + lam * g

x = ep.fista(objective, x0, iters=300)
```

## How it works

- The columns of $G(x)$ are the gradient fields $f_k^\top b_a(f_k \star x)$ of the basis potentials.
- $A = E[G^\top G]$ pairs the overlap of two filters with the joint statistics of their responses.
  $d = E[G^\top t]$, with $t = (y - x)/\sigma^2$ for `"mse"`, $(y - m)/\sigma^2$ at the MMSE estimate
  $m$ for `"map"`; for `"score"`, $d$ is the mean divergence.
- Ridge and smoothness weights are chosen on held-out images by the same objective. Shape
  constraints turn the solve into a non-negative quadratic program.

## Examples

| Script | |
|---|---|
| `examples/denoise.py` | fit on BSD400, denoise Set12 and BSD68 |
| `examples/deblur.py` | deblurring with a MAP prior and `fista` |
| `examples/learn_filters.py` | learned filters against the DCT bank |

On one H200, BSD400 for training, noise std 25/255:

| DCT 7×7 bank (48 filters) | Fit | Set12 | BSD68 |
|---|---|---|---|
| `objective="mse"`, one step | 2.6 s | 28.99 dB | 28.16 dB |
| `objective="map"` | 5.8 s | 29.04 dB | 28.30 dB |
| `objective="map", refine=3` | 185 s | 29.24 dB | 28.43 dB |

`examples/deblur.py` (Gaussian blur of std 1.6 px, noise 2.55/255, weight noise², no tuning) takes Set12
from 25.11 dB to 28.75 dB. `examples/learn_filters.py` learns 16 filters in 20 s that beat the 48 DCT
filters at noise 12.75/255 (BSD68: 31.70 dB against 31.65 dB).

## Tests

```bash
pytest src/explicit_priors_tests
```

The figure above comes from `python doc/make_figures.py --data path/to/BSD`.

## Citing

See `CITATION.cff`, or the *Cite this repository* button on GitHub.
