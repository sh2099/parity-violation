import logging
import platform
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def get_unique_dir(base: Path) -> Path:
    """As before—bump suffix until free."""
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = base.with_name(f"{base.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def pick_device(
    use_cuda: bool = False,
    use_dml: bool = False,
    use_mps: bool = False,
) -> torch.device:
    """
    Choose the best available device according to user flags and availability.

    Priority:
      1) CUDA (NVIDIA GPUs)
      2) DirectML (Windows)
      3) MPS (macOS)
      4) CPU

    Parameters
    ----------
    use_cuda : bool
        Enable cuda if available.
    use_dml : bool
        Enable DirectML (Windows) if available.
    use_mps : bool
        Enable Apple's MPS (macOS) if available.

    Returns
    -------
    torch.device
    """
    # 1) CUDA
    if use_cuda and torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")

    # 2) DirectML (Windows only)
    if use_dml and platform.system() == "Windows":
        try:
            import torch_directml

            dml_dev = torch_directml.device()
            logger.info("Using DirectML device: %s", dml_dev)
            return dml_dev
        except ImportError:
            logger.warning("torch-directml not installed; skipping DirectML")

    # 3) MPS (macOS only)
    if (
        use_mps
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        logger.info("Using MPS device")
        return torch.device("mps")

    # 4) Fallback to CPU
    logger.info("Falling back to CPU")
    return torch.device("cpu")
