API Reference
=============

Images are PyTorch tensors with values in [0, 1], of shape (H, W) or (B, H, W), on CPU or GPU.


Pretrained Priors
-----------------

.. automodule:: explicit_priors.pretrained
   :no-members:
   :no-index:

.. autofunction:: explicit_priors.pretrained

.. autofunction:: explicit_priors.available


Fitting
-------

.. autofunction:: explicit_priors.fit

.. autofunction:: explicit_priors.learn_filters


Model
-----

.. autoclass:: explicit_priors.FoE
   :members: energy, grad, energy_grad, responses, psi, phi, potentials, save, load


Denoising and Inverse Problems
------------------------------

.. autofunction:: explicit_priors.denoise

.. autofunction:: explicit_priors.one_step

.. autofunction:: explicit_priors.map_estimate

.. autofunction:: explicit_priors.fista


Filter Banks
------------

.. autofunction:: explicit_priors.filters.dct

.. autofunction:: explicit_priors.filters.gabor

.. autofunction:: explicit_priors.filters.gaussian_derivatives

.. autofunction:: explicit_priors.filters.laplacian_of_gaussian

.. autofunction:: explicit_priors.filters.pca

.. autofunction:: explicit_priors.filters.ica

.. autofunction:: explicit_priors.filters.symmetrize

.. autofunction:: explicit_priors.filters.stack

.. autofunction:: explicit_priors.filters.dilate

.. autofunction:: explicit_priors.filters.normalize


Potential Bases
---------------

.. autoclass:: explicit_priors.Hat

.. autoclass:: explicit_priors.Gauss


Normal Equations
----------------

.. autofunction:: explicit_priors.normal_equations
