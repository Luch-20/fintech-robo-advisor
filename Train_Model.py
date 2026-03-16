"""
Compatibility entrypoint.

The original docs reference `Train_Model.py` at project root. The actual training
script lives in `src/train_model.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"

    # Make sure `train_model.py` and its sibling modules can be imported.
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(project_root))

    from train_model import train_all_models  # noqa: WPS433

    train_all_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

