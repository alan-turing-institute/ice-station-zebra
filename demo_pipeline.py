# %% [markdown]
# # Ice Station Zebra Pipeline Demo
# 
# This demonstration showcases the complete Ice Station Zebra ML pipeline capabilities through CLI commands. 
# 
# **Target Audience:** Developer teams and future team members who want to understand our design decisions, 
# trade-offs, and flexible experimentation capabilities.
# 
# **You'll learn how to:**
# - Run our training pipeline end-to-endin three lines of code
# - Swap between different modelling paradigms
# - Reproduce runs and inspect the outputs
# - Evaluate the performance of the models in line with community standards on sea ice forecasting


# %% [markdown]
# ## Demo Structure
# 
# **Section 1: End-to-End Training**
# - Run a full zebra pipeline end2end using a minimal configuration & data
# - Inspect training artifacts and see evaluation outputs
# 
# **Section 2: Model Flexibility**
# - Switch between Encode-Process-Decode paradigm and standalone persistence model
# - Explore Encoder module functionality (Multimodality)
# 
# **Section 3: Evaluation Framework**
# - Evaluate and compare model performance using a pretrained model checkpoint
# - Explore different plotting formats and metrics
#
# **Section 4: Train it yourself - Advanced Example**
# - use anemoi functionality to fetch and inspect standard datsets
# - write your own config to train a model on a full dataset
# - see our pipeline data checks and validation in action

# %% [markdown]
# # Section 1: End-to-End Training Pipeline
# 
# In this section, we'll demonstrate the complete training pipeline using a simple **Naive encoder model** trained on a few days of data.
# For the purpose of this notebook we have created a minimal config file and uploaded some small subset of the data.
# The dataset contains a few days of sea ice concentration data (OSISAF) and corresponding atmospheric data (ERA5).
# We don't expect the model to do well, but it will give us a sense of the pipeline.

# %%
# Environment Verification
# Let's verify that our zebra cli tools are available and working

!uv --version

# %%
!uv run zebra --help


# %%
!uv run zebra datasets inspect --config-name=demo

# %%
!uv run zebra train --config-name=demo

# %%
!uv run zebra evaluate --config-name=demo

# %% [markdown]
# # Section 2: Model Flexibility
# 
# In this section, we'll demonstrate how easy it is to switch between different model architectures.
# We'll show the difference between standalone models and the encode-process-decode paradigm.

# %%
# TODO: Add model swapping demonstration
!uv run zebra train --config-name=demo ++train=persistence ++model=persistence

# %% [markdown]
# # Section 3: Evaluation Framework
# 
# Here we'll dive deep into the evaluation capabilities, comparing different models
# and exploring various plotting formats and metrics.
# For more interesting visualisations we will load a pretrained model checkpoint.

!uv run zebra evaluate --config-name=demo --checkpoint PATH_TO_CHECKPOINT


# %%
# TODO: Add evaluation framework demonstration

# %% [markdown]
# # Section 4: Train it yourself - Advanced Example
# 
# This section shows how to use this pipeline on your own data. Our pipeline builds on Anemoi functionality to fetch and inspect standard datasets,
# write your own config, and see our pipeline data checks and validation in action.

# %%
# Configuration Management with Hydra
# Following the README instructions, we'll create a local config file that inherits from base.yaml
# This demonstrates Zebra's config-driven approach and Hydra's inheritance system

# First, let's see what the default base path is configured to
!cat ice_station_zebra/config/base.yaml

# %%
# Let's examine our local configuration file
# This file inherits from base.yaml and overrides the base_path for local development
# Following the README instructions for creating local configs

!cat ice_station_zebra/config/demo.yaml
