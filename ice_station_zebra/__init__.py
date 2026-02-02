# We set this environment variable here so that it is set before we load torch
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
