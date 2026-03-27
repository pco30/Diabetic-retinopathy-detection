"""Backward-compatible entry point.

Use `python diabetic.py` to run the refactored training/evaluation pipeline.
"""

from drd.pipeline import main


if __name__ == "__main__":
    main()