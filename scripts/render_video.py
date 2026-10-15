"""Render a saved recovery tape from a real initial condition."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.render_video import cli

if __name__ == "__main__":
    cli()
