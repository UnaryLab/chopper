"""Shared cache for loaded data files."""

import pandas as pd
from typing import Dict

# Global cache for pickle files
_pickle_cache: Dict[str, pd.DataFrame] = {}


def load_pickle(path: str) -> pd.DataFrame:
    """Load a pickle file with caching.

    If the file has been loaded before, returns the cached version.
    Otherwise loads the file, caches it, and returns it.

    Args:
        path: Path to pickle file

    Returns:
        Loaded DataFrame
    """
    if path not in _pickle_cache:
        _pickle_cache[path] = pd.read_pickle(path)
    return _pickle_cache[path]


def clear_cache():
    """Clear the pickle cache."""
    _pickle_cache.clear()
