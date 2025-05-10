import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from rich.tree import Tree

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
    """
    if use_dml and platform.system() == "Windows":
        try:
            import torch_directml

            dml_dev = torch_directml.device()
            logger.info("Using DirectML device: %s", dml_dev)
            return dml_dev
        except ImportError:
            logger.warning("torch-directml not installed; skipping DirectML")
    """
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


def dict_to_tree(
    data: dict | DictConfig, tree: Tree = None, name: str = "Config", **tree_kwargs
) -> Tree:
    """Convert a nested dictionary to a Rich Tree."""
    if tree is None:
        tree = Tree(name, **tree_kwargs)

    for key, value in data.items():
        if isinstance(value, (dict, DictConfig)):
            dict_to_tree(value, tree.add(f"[cyan]{key}[/cyan]"))
        else:
            tree.add(f"[yellow]{key}[/yellow]: {value}")

    return tree
