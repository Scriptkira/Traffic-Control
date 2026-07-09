"""GPU/CUDA device resolution — single source of truth for which device
model backends run on. Never silently fall back to CPU without logging it.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(component: str) -> str:
    """Return 'cuda' or 'cpu' for the given component, logging the decision."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"{component}: CUDA available — using GPU ({name})")
        return "cuda"
    logger.warning(f"{component}: CUDA NOT available — falling back to CPU (will be slow)")
    return "cpu"
