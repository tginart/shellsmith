"""Allow the frozen Studio runtime to use ``python -m shellsmith``."""

from shellsmith.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
