"""Run native MLX setup using the selected Studio interpreter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_setup import main

if __name__ == "__main__":
    raise SystemExit(main())
