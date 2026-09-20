"""
Memory management utilities for constrained container environments (e.g. Render 512MB RAM).
"""

import ctypes
import gc
import logging
import sys

logger = logging.getLogger(__name__)


def release_memory() -> None:
    """
    Force garbage collection and release freed C++ heap arenas back to the OS.
    
    Python's standard gc.collect() frees Python references, but Linux glibc's
    memory allocator holds onto pages in its heap arena instead of returning
    them to the kernel. Calling malloc_trim(0) explicitly instructs glibc
    to release those pages, immediately lowering container RSS memory.
    """
    try:
        collected = gc.collect()
        if sys.platform != "win32":
            try:
                libc = ctypes.CDLL("libc.so.6")
                # malloc_trim(0) releases free memory from the top of the heap to the OS
                res = libc.malloc_trim(0)
                logger.debug(f"malloc_trim(0) executed, result: {res}, gc collected: {collected}")
            except Exception as trim_err:
                logger.debug(f"malloc_trim not available: {trim_err}")
    except Exception as e:
        logger.debug(f"Error releasing memory: {e}")
