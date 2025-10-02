Processors
==========

The processors module provides classes for processing data in latent space within the encode-process-decode pipeline.

Classes
-------

BaseProcessor
~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.processors.base_processor.BaseProcessor
      :members:
      :undoc-members:
      :show-inheritance:

UNetProcessor
~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.processors.unet.UNetProcessor
      :members:
      :undoc-members:
      :show-inheritance:

NullProcessor
~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.processors.null.NullProcessor
      :members:
      :undoc-members:
      :show-inheritance:

VitProcessor
~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.processors.vit.VitProcessor
      :members:
      :undoc-members:
      :show-inheritance:

DDPMProcessor
~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.models.processors.ddpm.DDPMProcessor
      :members:
      :undoc-members:
      :show-inheritance:

   .. note::
      This processor wraps the diffusion models from the :doc:`diffusion models <models-diffusion>` section.
      See :class:`GaussianDiffusion` and :class:`UNetDiffusion` for the underlying denoising algorithms.
