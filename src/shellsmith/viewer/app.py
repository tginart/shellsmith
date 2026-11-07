"""Serve the read-only shellsmith rollout dashboard.

Architecture map
----------------
The Flask layer has only three jobs: serve the static single-page dashboard, return
catalog JSON, and return one normalized run.  All filesystem traversal and artifact
interpretation lives in ``loader.py``.  The simulator never imports this package,
so viewing a run cannot affect an episode in progress.

Run locally:
    python -m shellsmith.viewer.app

Canonical artifact root (covers episode, campaign and training defaults):
    <system temp>/shellsmith-<uid>/ on POSIX (episodes/, campaigns/, runs/)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from shellsmith.paths import DEFAULT_LOG_ROOT

from .loader import load_catalog, load_run


def default_runs_root() -> Path:
    """Default Observatory root: same tree every campaign/run_scripts recipe writes.

    Override with SHELLSMITH_RUNS_ROOT or --runs-root when browsing an
    older path (for example checked-in artifacts/hero samples).
    """
    configured = os.environ.get("SHELLSMITH_RUNS_ROOT")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_LOG_ROOT


def _print_catalog(catalog: dict) -> None:
    """Print a short human inventory of discovered rollouts."""
    error = catalog.get("error")
    if error:
        print(f"Catalog error: {error}", flush=True)
        return
    runs = catalog.get("runs") or []
    print(f"Found {len(runs)} rollout(s)", flush=True)
    if not runs:
        print("  (none — episode.json not found under this root yet)", flush=True)
        return
    for run in runs:
        run_id = run.get("id", "?")
        status = run.get("status", "unknown")
        actions = run.get("action_count", 0)
        reward = run.get("reward")
        reward_text = f" reward={reward}" if reward is not None else ""
        err_count = len(run.get("errors") or [])
        err_text = f" errors={err_count}" if err_count else ""
        print(
            f"  {run_id}  status={status}  turns={actions}{reward_text}{err_text}",
            flush=True,
        )


def create_app(*, runs_root: Path) -> Flask:
    """Create a local, read-only dashboard bound to one artifact root.

    Main callers:
        ``main`` and offline Flask tests.

    The root is injected rather than global so tests and local experiments cannot
    accidentally read an unrelated run directory.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():  # type: ignore[no-untyped-def]
        return render_template("index.html", runs_root=str(runs_root.resolve()))

    @app.get("/api/runs")
    def runs():  # type: ignore[no-untyped-def]
        catalog = load_catalog(runs_root)
        run_list = catalog.get("runs") or []
        print(f"GET /api/runs -> {len(run_list)} rollout(s)", flush=True)
        if catalog.get("error"):
            print(f"  catalog error: {catalog['error']}", flush=True)
        return jsonify(catalog)

    @app.get("/api/runs/<path:run_id>")
    def run_detail(run_id: str):  # type: ignore[no-untyped-def]
        try:
            record, error = load_run(runs_root, run_id)
        except Exception as exc:
            print(f"GET /api/runs/{run_id} -> FAILED: {exc}", flush=True)
            return jsonify({"error": f"could not load rollout detail: {exc}"}), 500
        if error:
            print(f"GET /api/runs/{run_id} -> 404: {error}", flush=True)
            return jsonify({"error": error}), 404
        summary = (record or {}).get("summary") or {}
        turns = (record or {}).get("turns") or []
        status = summary.get("status", "unknown")
        reward = summary.get("reward")
        reward_text = f" reward={reward}" if reward is not None else ""
        print(
            f"GET /api/runs/{run_id} -> status={status} "
            f"turns={len(turns)}{reward_text}",
            flush=True,
        )
        return jsonify(record)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only shellsmith rollout dashboard")
    parser.add_argument("--runs-root", default=str(default_runs_root()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8734)
    args = parser.parse_args(argv)
    root = Path(args.runs_root).expanduser()

    # Quiet Flask/werkzeug access lines; we print our own status instead.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    def say(message: str) -> None:
        print(message, flush=True)

    say(f"Viewer:  http://{args.host}:{args.port}/")
    say(f"Root:    {root.resolve()}")
    if not root.is_dir():
        say(f"ERROR: runs root does not exist: {root.resolve()}")
        return 1
    catalog = load_catalog(root)
    _print_catalog(catalog)
    say("Serving. Open the URL above; Ctrl+C to stop.")
    try:
        create_app(runs_root=root).run(
            host=args.host, port=args.port, debug=False, threaded=True
        )
    except OSError as error:
        say(f"ERROR: could not bind {args.host}:{args.port}: {error}")
        say("Another viewer is probably still running. Kill it or pick --port.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
