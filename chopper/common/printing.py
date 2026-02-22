"""Colored console output utilities for logging and debugging."""

from sys import stderr


def fancyprint(msg: str, dim: bool, bold: bool, type: str, color: str) -> None:
    """Print colored and styled message to stderr.
    
    Args:
        msg: Message string to print
        dim: If True, apply dim text style
        bold: If True, apply bold text style
        type: Type prefix (e.g., "ERR: ", "WARN:")
        color: ANSI color code (e.g., "31" for red)
    """
    styles = ""
    if bold:
        styles += "\033[1m"
    if dim:
        styles += "\033[2m"
    print(
        f"\033[1;{color}m{type} \033[0m{styles} {msg}\033[0m", file=stderr)


def err(msg: str, dim: bool = False, bold: bool = False) -> None:
    """Print error message in red.
    
    Args:
        msg: Error message to print
        dim: If True, apply dim text style
        bold: If True, apply bold text style
    """
    fancyprint(msg, dim, bold, "ERR: ", "31")


def warn(msg: str, dim: bool = False, bold: bool = False) -> None:
    """Print warning message in yellow.
    
    Args:
        msg: Warning message to print
        dim: If True, apply dim text style
        bold: If True, apply bold text style
    """
    fancyprint(msg, dim, bold, "WARN:", "33")


def info(msg: str, dim: bool = False, bold: bool = False) -> None:
    """Print info message in blue.
    
    Args:
        msg: Info message to print
        dim: If True, apply dim text style
        bold: If True, apply bold text style
    """
    fancyprint(msg, dim, bold, "INFO:", "34")


def output(msg: str, dim: bool = False, bold: bool = False) -> None:
    """Print output message in green.
    
    Args:
        msg: Output message to print
        dim: If True, apply dim text style
        bold: If True, apply bold text style
    """
    fancyprint(msg, dim, bold, "OUT: ", "32")


def kern_name_short(name: str, length: int = 80) -> str:
    """Truncate kernel name to specified length with ellipsis.
    
    Args:
        name: Kernel name to truncate
        length: Maximum length before truncation
        
    Returns:
        Truncated kernel name with '...' appended if truncated
    """
    return f"{name[:length]}{'...' if len(name) > length else ''}"
