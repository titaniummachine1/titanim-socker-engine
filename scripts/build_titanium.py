"""Build real-game Titanium.txt via AIGamePyLibrary.

Thin entry point — all the actual logic lives in the `titanium` package
next to this file (geometry, ball_physics, shot, tackle, carrier,
goalkeeper, debug_viz, graph, deploy). This script just wires: build the
graph, write it out, promote it if asked.

CANONICAL LOCATION. This file is the competition engine and lives in the
titanim-socker-engine repo. Both repos are public now, so there is no secrecy
rule here any more — but keep the two trees from bleeding into each other:

  * ENGINE SOURCE lives here, and only here. Do not copy modules into
    aicomp-soccer-sim; that repo is the simulator and its harnesses.
  * BUILT GRAPHS are artifacts, not source. A single graph is 5-16 MB of JSON
    and is rewritten by every build, so `data/titanium/` stays gitignored in
    the sim tree and `out/ti_*.txt` stays gitignored here. Committing them is
    churn, not history.

Building writes the graph to the live Unity save folder and to the sim's
`data/titanium/` so the drill harnesses can load it via
`--gk data/titanium/Titanium.txt`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from titanium import deploy, graph  # noqa: E402
from titanium import profile as _profile  # noqa: E402


def main() -> None:
    # --profile must wrap the node constructor BEFORE anything is built.
    profiling = "--profile" in sys.argv
    if profiling:
        _profile.install()

    graph.build()

    if profiling:
        from titanium._env import graph_data

        # Report the graph the build actually writes, i.e. after SaveData's
        # optimiser has run — profiling the pre-optimise graph would blame
        # modules for nodes that never ship.
        deploy.write_candidate()
        _profile.report(
            graph_data["serializableNodes"],
            graph_data["serializableConnections"],
            write_baseline="--baseline" in sys.argv,
        )
    else:
        deploy.write_candidate()

    deploy.promote_if_requested()


if __name__ == "__main__":
    main()
