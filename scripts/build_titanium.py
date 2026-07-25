"""Build real-game Titanium.txt via AIGamePyLibrary.

Thin entry point — all the actual logic lives in the `titanium` package
next to this file (geometry, ball_physics, shot, tackle, carrier,
goalkeeper, debug_viz, graph, deploy). This script just wires: build the
graph, write it out, promote it if asked.

CANONICAL LOCATION. This file is the competition engine and lives in the
PRIVATE titanim-socker-engine repo. It must never be committed to
aicomp-soccer-sim, which is a PUBLIC repo (that tree ignores it explicitly).
Building writes the graph out to two places: the live Unity save folder, and
the public sim's gitignored `data/titanium/` so the drill harnesses can load
it via `--gk data/titanium/Titanium.txt` without the source ever going with
it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from titanium import deploy, graph  # noqa: E402


def main() -> None:
    graph.build()
    deploy.write_candidate()
    deploy.promote_if_requested()


if __name__ == "__main__":
    main()
