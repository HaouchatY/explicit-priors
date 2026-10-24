Explicit Priors
===============

.. toctree::
   :maxdepth: 4
   :hidden:

   api
   references


.. |gh-badge| image:: https://img.shields.io/badge/GitHub-Repo-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/HaouchatY/explicit-priors

.. |docs-badge| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github
   :alt: Documentation
   :target: https://HaouchatY.github.io/explicit-priors/

|gh-badge| |docs-badge|


Explicit Priors fits fields-of-experts image priors in closed form: one linear solve, no training loop.

.. image:: potentials.png
   :width: 100%
   :alt: 3x3 DCT filters and their MAP potentials


Method
------

A fields of experts :cite:`roth2009fields` scores an image :math:`\bbx` through the responses of filters
:math:`\bbf_{k}`,

.. math::
   :label: foe

   \Phi(\bbx) = \sum_{k=1}^{K} \sum_{i} \phi_{k}\big( (\bbf_{k} \star \bbx)_{i} \big),
   \qquad
   \phi_{k}' = \sum_{a} c_{ka} \, b_{a},

where the potentials :math:`\phi_{k}` are expanded on a fixed basis :math:`b_{a}`.
The gradient :math:`\nabla \Phi(\bbx) = \bbG(\bbx) \, \bbc` is linear in the coefficients, so fitting
:math:`\Phi` to images is a least-squares problem in :math:`\bbc`, solved by its normal equations

.. math::

   \bbA \bbc = \bbd, \qquad \bbA = \bE\big[ \bbG^{\top} \bbG \big], \qquad \bbd = \bE\big[ \bbG^{\top} \mathbf{t} \big].

The points at which :math:`\bbG` is evaluated and the target :math:`\mathbf{t}` set the objective, with noisy images
:math:`\bby = \bbx + \sigma \mathbf{n}`:

- *one-step denoiser* :math:`\bby - \sigma^{2} \nabla \Phi(\bby)` with the lowest error: points :math:`\bby`, target
  :math:`(\bby - \bbx) / \sigma^{2}` (denoising score matching :cite:`vincent2011connection`);
- *MAP denoiser* :math:`\argmin_{\bbx} \tfrac12 \norm{\bbx - \bby}^{2} + \sigma^{2} \Phi(\bbx)` that returns the MMSE
  estimate :math:`\bbm` :cite:`gribonval2011should`: points :math:`\bbm`, target :math:`(\bby - \bbm) / \sigma^{2}`.
  Gauss-Newton steps then fit the MAP output itself;
- *prior of clean images*, by score matching :cite:`hyvarinen2005estimation`.

The filters can be analytic, computed from patches in closed form, or learned with the potentials by variable
projection :cite:`golub1973differentiation`: the potentials are re-solved at every step of the filter descent.


Usage
-----

.. code-block:: python

   import explicit_priors as ep

   prior = ep.fit(ep.filters.dct(7), images, sigma=25 / 255, objective="map")
   x = ep.denoise(prior, y)


Setup
-----

.. code-block:: bash

   # user install
   pip install explicit_priors@git+https://github.com/HaouchatY/explicit-priors.git

   # developer install
   git clone https://github.com/HaouchatY/explicit-priors.git
   cd explicit-priors/
   pip install -e ".[dev,examples]"
   pre-commit install

   # run test suite
   pytest src/explicit_priors_tests
