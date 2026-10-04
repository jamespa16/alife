"""alife - evolving creature world

Setup:  uv sync           (or: pip install torch pygame numpy)
Run:    uv run alife      (or: python -m alife; --help for headless runs, seeds, CSV logs)
Keys:   SPACE pause | C colour mode (lineage / loop count) | UP/DOWN sim steps per frame

Everything lives in fixed-size GPU tensors of length N_MAX with an `alive` mask.
Each creature has a weight-tied recurrent brain; its loop count is a gene that
costs energy per pass. There is no fitness function: eat, move, split, die.

Modules: config (constants) | brain (recurrent net) | world (simulation step)
         render (frame) | app (CLI, logging, headless and pygame loops)
"""

from .app import main

__all__ = ["main"]
