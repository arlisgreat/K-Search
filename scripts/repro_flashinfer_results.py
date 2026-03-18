#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages

SYSTEM_PREFIX = {
    "ksearch": "ksearch_best",
    "oe": "oe_best",
    "shinka": "shinka_best",
}


def _resolve_result_dir(repo_root: Path, definition: str) -> Path:
    matches = sorted((repo_root / "results").glob(f"*/*{definition}"))
    if not matches:
        raise FileNotFoundError(f"Could not find a results directory for definition '{definition}'")
    if len(matches) > 1:
        exact = [path for path in matches if path.name == definition]
        if len(exact) == 1:
            return exact[0]
    return matches[0]


def _load_solution(result_dir: Path, definition: str, system: str) -> Solution:
    prefix = SYSTEM_PREFIX[system]
    cu_files = sorted(result_dir.glob(f"{prefix}*.cu"))
    h_files = sorted(result_dir.glob(f"{prefix}*.h"))
    cpp_files = sorted(result_dir.glob(f"{prefix}*.cpp"))
    if len(cu_files) != 1 or len(h_files) != 1 or len(cpp_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .cu/.h/.cpp trio for system='{system}' under {result_dir}"
        )
    sources = [
        SourceFile(path="kernel.h", content=h_files[0].read_text(encoding="utf-8")),
        SourceFile(path="kernel.cu", content=cu_files[0].read_text(encoding="utf-8")),
        SourceFile(path="main.cpp", content=cpp_files[0].read_text(encoding="utf-8")),
    ]
    return Solution(
        name=f"{system}_{definition}",
        definition=definition,
        author=system,
        spec=BuildSpec(
            language=SupportedLanguages.CUDA,
            target_hardware=[],
            entry_point="main.cpp::run",
        ),
        sources=sources,
        description=f"Published {system} kernel from K-Search results/{result_dir.relative_to(result_dir.parents[1])}",
    )


def main() -> None:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description="Benchmark published FlashInfer result kernels.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data" / "flashinfer-trace",
        help="Path to the flashinfer-trace dataset inside this repo.",
    )
    parser.add_argument("--definition", required=True, help="FlashInfer definition to benchmark.")
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=sorted(SYSTEM_PREFIX.keys()),
        default=["ksearch", "oe", "shinka"],
        help="Which published systems to benchmark.",
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--use-isolated-runner", action="store_true")
    parser.add_argument("--parallel-workloads", action="store_true")
    parser.add_argument("--max-parallel-workloads", type=int, default=0)
    parser.add_argument("--dump-traces", action="store_true")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=repo_root / ".ksearch-output" / "repro_flashinfer",
        help="Directory for the saved report JSON.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    result_dir = _resolve_result_dir(repo_root, args.definition)
    solutions = [_load_solution(result_dir, args.definition, system) for system in args.systems]

    task = FlashInferBenchTask.from_cli_args(
        task_path=str(dataset_root),
        definition_name=str(args.definition),
        warmup_runs=args.warmup_runs,
        iterations=args.iterations,
        num_trials=args.num_trials,
        rtol=args.rtol,
        atol=args.atol,
        use_isolated_runner=bool(args.use_isolated_runner),
        parallel_workloads=bool(args.parallel_workloads),
        max_parallel_workloads=args.max_parallel_workloads,
        baseline_solution=None,
        feedback_workloads=None,
        feedback_trace_policy="first",
        num_feedback_workloads=1,
        artifacts_dir=str(args.artifacts_dir),
    )
    report = task.run_final_evaluation(
        solutions=solutions,
        dump_traces=bool(args.dump_traces),
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.artifacts_dir.expanduser().resolve() / args.definition / f"report_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved JSON report to {out_path}")


if __name__ == "__main__":
    main()
