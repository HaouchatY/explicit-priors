# Explicit priors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fields-of-experts image priors whose potentials are fitted by one linear solve.

```math
\Phi(x) = \sum_k \sum_i \phi_k\big((f_k \star x)_i\big)
```

The gradient of Φ is linear in the coefficients of the potentials φ<sub>k</sub>, so fitting them to
images is a least-squares problem: one solve, no training loop.

<p align="center"><img src="doc/potentials.png" width="720" alt="DCT 3x3 filters and their potentials"></p>
<p align="center"><sub>3×3 DCT filters and their MAP potentials, noise 25/255, 400 BSD images:
<code>python doc/make_figures.py --data path/to/BSD</code></sub></p>

## Installation

```bash
pip install "explicit_priors@git+https://github.com/HaouchatY/explicit-priors.git"
```

## Usage

```python
import explicit_priors as ep

prior = ep.fit(ep.filters.dct(7), images, sigma=25 / 255, objective="map")
x = ep.denoise(prior, y)
```

| Argument of `fit` | Values |
|---|---|
| `filters` | `ep.filters.dct`, `gabor`, `gaussian_derivatives`, `laplacian_of_gaussian`, `pca`, `ica`, `ep.learn_filters`, or any (K, s, s) tensor |
| `objective` | `"mse"`: one-step denoiser. `"map"`: MAP denoiser. `"score"`: prior of clean images |
| `shape` | `"free"`, `"monotone"`, `"convex"` |
| `refine` | Gauss–Newton steps on the MAP denoiser |

For inverse problems, minimise your data term plus `lam * prior.energy(x)` with `ep.fista`
(`examples/deblur.py`).

## Examples

`examples/denoise.py`, `examples/deblur.py`, `examples/learn_filters.py`

## Tests

```bash
pytest src/explicit_priors_tests
```
