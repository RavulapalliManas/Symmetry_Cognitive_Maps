"""Prepare experiments, replay a reference, and run bounded X1 CPU development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from map_dynamics.preparation import audit_legacy, load_yaml, prepare, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Directory containing the package's configs and registries")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Check preparation consistency; does not verify scientific invariants")
    commands.add_parser("needs", help="Show unresolved requirements before scientific work")
    draft = commands.add_parser("prepare", help="Export a draft plan; never launch a job")
    draft.add_argument("--experiment", required=True, choices=["X0", "X1", "X2", "X3", "X4", "X5"])
    draft.add_argument("--output", type=Path, help="New JSON file on the fleet; existing files are refused")
    audit = commands.add_parser("audit-legacy", help="Inventory source files without importing models")
    audit.add_argument("--source-root", type=Path, required=True)
    replay = commands.add_parser("replay-legacy", help="Evaluate pinned legacy checkpoints on paired diagnostic inputs; no training")
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    development = commands.add_parser("develop-x1", help="Bounded CPU engineering run; never launch a scientific campaign")
    development.add_argument("--config", type=Path, default=Path("configs/protocols/X1_development.yaml"))
    development.add_argument("--output", type=Path, required=True)
    development.add_argument("--resume", type=Path)
    development.add_argument("--stop-after", type=int, help="Stop at this total rollout-update count without changing the configured budget")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate(args.root)
        elif args.command == "needs":
            result = load_yaml(args.root / "registries/preparation.yaml")
        elif args.command == "audit-legacy":
            result = audit_legacy(args.source_root)
        elif args.command == "develop-x1":
            from map_dynamics.learning import run_development
            result = run_development(args.root, load_yaml(args.root / args.config), args.output,
                                     resume=args.resume, stop_after=args.stop_after)
        elif args.command == "replay-legacy":
            from map_dynamics.replay import run
            config = load_yaml(args.root / args.config)
            if not Path(config["legacy_root"]).expanduser().is_absolute():
                config["legacy_root"] = str((args.root / config["legacy_root"]).resolve())
            report = run(config, args.output)
            result = {"status": report["status"], "report": str(args.output / "report.json"), "scientific_hypotheses_supported": False}
        else:
            result = prepare(args.root, args.experiment)
        content = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.command == "prepare" and args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(f"Draft written: {args.output}. No job launched.")
        else:
            print(content, end="")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(2, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
