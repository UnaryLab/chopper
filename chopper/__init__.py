import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="\U0001f415 <level>{level:<8}</level> | {time:HH:mm:ss} | {message}",
)
