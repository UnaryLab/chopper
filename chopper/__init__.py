"""
# Chopper

GPU characterization tool for analyzing multi-GPU performance during
distributed deep learning workloads.

This is Chopper:

![Chopper](chopper.jpg)
"""

__all__ = ["common", "plots", "profile"]

import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="\U0001f415 <level>{level:<8}</level> | {time:HH:mm:ss} | {message}",
)
