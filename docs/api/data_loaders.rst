Data Loaders
============

The data loaders module provides classes for loading and managing datasets in the Ice Station Zebra framework.

Classes
-------

CombinedDataset
~~~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.data_loaders.combined_dataset.CombinedDataset
      :members:
      :undoc-members:
      :show-inheritance:

ZebraDataModule
~~~~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.data_loaders.zebra_data_module.ZebraDataModule
      :members:
      :undoc-members:
      :show-inheritance:

ZebraDataset
~~~~~~~~~~~~

.. container:: toggle

   .. autoclass:: ice_station_zebra.data_loaders.zebra_dataset.ZebraDataset
      :members:
      :undoc-members:
      :show-inheritance:

Usage Examples
--------------

Loading a Dataset
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ice_station_zebra.data_loaders.zebra_dataset import ZebraDataset

   # Load a dataset
   dataset = ZebraDataset("path/to/dataset.zarr")

   # Access data
   data = dataset[0]  # Get first sample


Combining Multiple Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ice_station_zebra.data_loaders.combined_dataset import CombinedDataset

   # Create a combined dataset from multiple ZebraDatasets
   combined = CombinedDataset(
       datasets=[dataset1, dataset2, dataset3],
       target="target_dataset_name",
       n_forecast_steps=4,
       n_history_steps=3
   )

   # Access combined data
   sample = combined[0]  # Returns dict with input and target data

Using ZebraDataModule
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ice_station_zebra.data_loaders.zebra_data_module import ZebraDataModule
   from omegaconf import DictConfig

   # Initialize with configuration
   data_module = ZebraDataModule(config)

   # Get data loaders
   train_loader = data_module.train_dataloader()
   val_loader = data_module.val_dataloader()
   test_loader = data_module.test_dataloader()
