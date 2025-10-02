Diffusion Models
================

.. note::
   Diffusion models provide the underlying denoising algorithms. To use them within the encode–process–decode pipeline, wrap them via :class:`DDPMProcessor` (see :doc:`Processors <models-processors>`).

Classes
-------

GaussianDiffusion
~~~~~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.diffusion.gaussian_diffusion.GaussianDiffusion
      :members:
      :undoc-members:
      :show-inheritance:

UNetDiffusion
~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.diffusion.unet_diffusion.UNetDiffusion
      :members:
      :undoc-members:
      :show-inheritance:
