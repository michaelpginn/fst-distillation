import sys
from unittest import mock

try:
    import wandb
except ImportError:
    print("`wandb` not found, disabling logging.")
    # Create a mock wandb module
    wandb = mock.MagicMock(name="wandb")

    # Insert the mock wandb module into sys.modules
    sys.modules["wandb"] = wandb

__all__ = ["wandb"]
