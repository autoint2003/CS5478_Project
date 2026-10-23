"""Dataset generation entry. Forwards to the six-branch expert pipeline."""

from data_generation.expert_pipeline.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
