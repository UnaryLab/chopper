from sys import stderr


def fancyprint(msg: str, dim: bool, bold: bool, type: str, color: str) -> None:
    styles = ""
    if bold:
        styles += "\033[1m"
    if dim:
        styles += "\033[2m"
    print(
        f"\033[1;{color}m{type} \033[0m{styles} {msg}\033[0m", file=stderr)


def err(msg: str, dim: bool = False, bold: bool = False) -> None:
    fancyprint(msg, dim, bold, "ERR: ", "31")


def warn(msg: str, dim: bool = False, bold: bool = False) -> None:
    fancyprint(msg, dim, bold, "WARN:", "33")


def info(msg: str, dim: bool = False, bold: bool = False) -> None:
    fancyprint(msg, dim, bold, "INFO:", "34")


def output(msg: str, dim: bool = False, bold: bool = False) -> None:
    fancyprint(msg, dim, bold, "OUT: ", "32")


def kern_name_short(name: str, length: int = 80) -> str:
    return f"{name[:length]}{'...' if len(name) > length else ''}"
