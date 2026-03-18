#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from k_search.utils.round_checkpoints import list_checkpoint_runs, resolve_round_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="List K-Search per-round checkpoints for a task.")
    parser.add_argument("--task-name", required=True, help="Task/definition name, e.g. gpumode_trimul or mla_paged_decode_h16_ckv512_kpe64_ps1")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint root directory (default: <repo>/checkpoints)")
    parser.add_argument("--run-id", default=None, help="If set, show details for a specific run")
    parser.add_argument("--round", type=int, default=None, help="If used with --run-id, inspect a specific round (default: latest)")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    if args.run_id:
        ref = resolve_round_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            task_name=args.task_name,
            run_id=args.run_id,
            round_num=args.round,
        )
        payload = {
            "task_name": args.task_name,
            "run_id": ref.run_id,
            "round_num": ref.round_num,
            "round_dir": str(ref.round_dir),
            "solution_path": str(ref.solution_path),
            "metadata_path": str(ref.metadata_path),
            "world_model_path": (str(ref.world_model_path) if ref.world_model_path is not None else None),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(f"run_id={payload['run_id']}")
        print(f"round={payload['round_num']}")
        print(f"round_dir={payload['round_dir']}")
        print(f"solution={payload['solution_path']}")
        if payload["world_model_path"]:
            print(f"world_model={payload['world_model_path']}")
        print(f"metadata={payload['metadata_path']}")
        return

    runs = list_checkpoint_runs(checkpoint_dir=args.checkpoint_dir, task_name=args.task_name)
    if args.json:
        print(json.dumps(runs, ensure_ascii=False, indent=2))
        return
    if not runs:
        print("No checkpoint runs found.")
        return
    for run in runs:
        print(
            "\t".join(
                [
                    str(run.get("run_id", "")),
                    f"latest_round={run.get('latest_round', '')}",
                    f"best_round={run.get('best_round', '')}",
                    f"best_score={run.get('best_score', '')}",
                    str(run.get("updated_at_utc", "")),
                ]
            )
        )


if __name__ == "__main__":
    main()
