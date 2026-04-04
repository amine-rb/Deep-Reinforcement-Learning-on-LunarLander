"""
Shared logging utilities used across all RL algorithms.

Provides a TeeLogger that mirrors stdout to a timestamped log file,
and helpers for printing structured configuration dumps.
"""

import os
import sys
from datetime import datetime
from typing import Tuple


class TeeLogger:
    """Mirrors stdout to both the terminal and a log file simultaneously."""

    def __init__(self, filepath: str, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filepath, mode, encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def setup_logging(log_dir: str, experiment_name: str) -> Tuple[str, TeeLogger]:
    """
    Initialize timestamped file logging and redirect stdout through TeeLogger.

    Args:
        log_dir: Directory where log files are written.
        experiment_name: Prefix for the log filename.

    Returns:
        Tuple of (log file path, TeeLogger instance).
    """
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{experiment_name}_{ts}.log")
    tee = TeeLogger(log_path)

    print("=" * 80)
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log     : {log_path}")
    print("=" * 80 + "\n")

    return log_path, tee


def log_config(cfg) -> None:
    """Print all fields of a dataclass Config object in a formatted table."""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for name in cfg.__dataclass_fields__:
        print(f"  {name:30s} = {getattr(cfg, name)}")
    print("=" * 80 + "\n")
