# API Reference

This section contains detailed documentation for all classes and functions in the Ice Station Zebra framework.

## Modules

### Data Loaders
Classes for loading and managing datasets:
- **CombinedDataset** - Combines multiple ZebraDatasets for training
- **ZebraDataModule** - Lightning DataModule for dataset management
- **ZebraDataset** - Base dataset class for individual datasets

### Data Processors
Classes for preprocessing and transforming data:
- **ZebraDataProcessor** - Main data processing pipeline
- **ZebraDataProcessorFactory** - Factory for creating processors

### Models
Neural network models and architectures:
- **[Main Models](models.md)** - Core model classes (ZebraModel, EncodeProcessDecode, Persistence)
- **[Common Components](models-common.md)** - Building blocks and utilities
- **[Encoders](models-encoders.md)** - Input encoding components
- **[Decoders](models-decoders.md)** - Output decoding components
- **[Processors](models-processors.md)** - Latent space processing components
- **[Diffusion Models](models-diffusion.md)** - Diffusion-based forecasting algorithms

### Training
Training utilities and trainers:
- **ZebraTrainer** - Main training class

### Evaluation
Evaluation metrics and utilities:
- **ZebraEvaluator** - Model evaluation class

### Types
Type definitions and data structures:
- **ArrayTCHW** - Time-Channel-Height-Width array type
- **DataSpace** - Data space definition
- **DataloaderArgs** - DataLoader arguments

### Visualisations
Plotting and visualization utilities:
- **PlottingCore** - Core plotting functionality
- **PlottingMaps** - Map-based visualizations
- **Layout** - Plot layout utilities
- **Convert** - Data conversion utilities
