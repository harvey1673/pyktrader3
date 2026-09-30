"""Compatibility wrapper for data dependency monitor.

Use misc_scripts/data_dependency_monitor.py for production runs.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc_scripts.data_dependency_monitor import main


if __name__ == "__main__":
    raise SystemExit(main())
