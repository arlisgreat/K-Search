#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from k_search.tasks.gpu_mode.evaluator import evaluate_trimul_submission

BM_LABELS = [
    "seq=256  bs=2 dim=128 nmsk=T norm",
    "seq=768  bs=1 dim=128 nmsk=T cchy",
    "seq=256  bs=2 dim=384 nmsk=F norm",
    "seq=512  bs=1 dim=128 nmsk=T norm",
    "seq=1024 bs=1 dim=128 nmsk=T cchy",
    "seq=768  bs=1 dim=384 nmsk=F norm",
    "seq=1024 bs=1 dim=384 nmsk=T norm",
]


def _detect_language(code: str) -> str:
    text = code or ""
    if "import triton" in text or "triton.jit" in text or "triton.language" in text:
        return "triton"
    return "python"


def _geo_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _format_stats(values: list[float | None]) -> tuple[str, str]:
    usable = [v for v in values if isinstance(v, (int, float))]
    if not usable:
        return "ERR", "ERR"
    avg = mean(usable)
    std = stdev(usable) if len(usable) > 1 else 0.0
    return f"{avg:8.3f} ms", f"{std:8.4f} ms"


def _candidate_files(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.glob("*.py") if path.is_file() and path.name != "__init__.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the published GPUMode TriMul result kernels.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "gpumode_trimul",
        help="Directory containing the published .py result kernels.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=REPO_ROOT / "k_search" / "tasks" / "gpu_mode" / "trimul",
        help="Vendored GPUMode TriMul task directory.",
    )
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated benchmark runs.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the aggregated report.",
    )
    parser.add_argument(
        "--keep-run-dirs",
        action="store_true",
        help="Keep per-run temp directories for debugging.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    task_dir = args.task_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else results_dir.parent.parent / ".ksearch-output" / "repro_gpumode_trimul.json"
    )

    candidates = _candidate_files(results_dir)
    if not candidates:
        raise SystemExit(f"No .py kernels found in {results_dir}")

    cache_root = Path(os.environ.get("TRITON_CACHE_DIR", results_dir.parent.parent / ".cache" / "triton"))
    tmp_root = Path(os.environ.get("TMPDIR", results_dir.parent.parent / ".tmp")) / "repro_gpumode_trimul"
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(candidates)} kernels: {' '.join(path.stem for path in candidates)}")
    print("")

    report: dict[str, dict[str, object]] = {}
    per_kernel_means: dict[str, list[list[float | None]]] = {}
    per_kernel_geos: dict[str, list[float | None]] = {}

    for path in candidates:
        name = path.stem
        code = path.read_text(encoding="utf-8")
        language = _detect_language(code)
        per_kernel_means[name] = []
        per_kernel_geos[name] = []

        print("=" * 60)
        print(f"  Kernel: {name}  ({args.num_runs} runs, language={language})")
        print("=" * 60)

        raw_runs: list[dict[str, object]] = []
        for run_idx in range(1, args.num_runs + 1):
            run_cache_dir = cache_root / f"{name}_run{run_idx}"
            run_tmp_dir = tmp_root / f"{name}_run{run_idx}"
            if run_cache_dir.exists():
                shutil.rmtree(run_cache_dir)
            run_cache_dir.mkdir(parents=True, exist_ok=True)
            run_tmp_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = str(run_cache_dir)

            summary = evaluate_trimul_submission(
                submission_code=code,
                mode="benchmark",
                language=language,
                task_dir=task_dir,
                keep_tmp=args.keep_run_dirs,
                tmpdir=run_tmp_dir,
                verbose=True,
            )

            per_bm_ms = [us / 1000.0 for us in summary.per_benchmark_means_us]
            while len(per_bm_ms) < len(BM_LABELS):
                per_bm_ms.append(None)
            geo_ms = _geo_mean([v for v in per_bm_ms if isinstance(v, (int, float))])

            per_kernel_means[name].append(per_bm_ms)
            per_kernel_geos[name].append(geo_ms)
            raw_runs.append(
                {
                    "run": run_idx,
                    "status": summary.status,
                    "latency_ms": summary.latency_ms,
                    "per_benchmark_ms": per_bm_ms,
                    "run_key": summary.run_key,
                    "run_success": summary.run_success,
                    "run_passed": summary.run_passed,
                }
            )
            print(f"  Run {run_idx} done")
            if not args.keep_run_dirs:
                shutil.rmtree(run_tmp_dir, ignore_errors=True)

        report[name] = {
            "language": language,
            "runs": raw_runs,
        }
        print("")

    col_w = max(max(len(name) for name in per_kernel_means) + 4, 26)
    total_w = 40 + 2 + col_w * len(per_kernel_means)

    print("=" * total_w)
    print(f"{f'PER-BENCHMARK RESULTS (mean +/- std across {args.num_runs} run(s), ms)':^{total_w}s}")
    print("=" * total_w)
    header = f"{'Benchmark':>40s}"
    for name in per_kernel_means:
        header += f"  {name:^{col_w}s}"
    print(header)
    print("-" * total_w)

    for bm_idx, label in enumerate(BM_LABELS):
        row = f"{label:>40s}"
        for name, runs in per_kernel_means.items():
            vals = [run[bm_idx] for run in runs if bm_idx < len(run)]
            usable = [v for v in vals if isinstance(v, (int, float))]
            if usable:
                avg = mean(usable)
                std = stdev(usable) if len(usable) > 1 else 0.0
                cell = f"{avg:8.3f} +/- {std:<5.3f} ms"
            else:
                cell = "ERR"
            row += f"  {cell:^{col_w}s}"
        print(row)

    print("")
    print("=" * total_w)
    print(f"{f'GEOMETRIC MEAN ACROSS {args.num_runs} RUN(S)':^{total_w}s}")
    print("=" * total_w)
    header = f"{'':>15s}"
    for name in per_kernel_geos:
        header += f"  {name:^{col_w}s}"
    print(header)
    sep_w = 15 + 2 + col_w * len(per_kernel_geos)
    print("-" * sep_w)

    for run_idx in range(args.num_runs):
        row = f"{('Run ' + str(run_idx + 1)):>15s}"
        for name, geos in per_kernel_geos.items():
            val = geos[run_idx]
            cell = f"{val:8.3f} ms" if isinstance(val, (int, float)) else "ERR"
            row += f"  {cell:^{col_w}s}"
        print(row)

    print("-" * sep_w)
    row_avg = f"{'Avg Geo Mean':>15s}"
    row_std = f"{'Std Geo Mean':>15s}"
    for name, geos in per_kernel_geos.items():
        usable = [v for v in geos if isinstance(v, (int, float))]
        if usable:
            avg = mean(usable)
            std = stdev(usable) if len(usable) > 1 else 0.0
            row_avg += f"  {f'{avg:8.3f} ms':^{col_w}s}"
            row_std += f"  {f'{std:8.4f} ms':^{col_w}s}"
        else:
            row_avg += f"  {'N/A':^{col_w}s}"
            row_std += f"  {'N/A':^{col_w}s}"
    print(row_avg)
    print(row_std)
    print("")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved JSON report to {output_json}")


if __name__ == "__main__":
    main()
