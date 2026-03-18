from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from k_search.tasks.task_base import EvalResult, Solution


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str, *, default: str) -> str:
    s = str(value or "").strip()
    if not s:
        return default
    out = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in s)
    return out or default


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stringify(value: Any) -> str:
    try:
        return str(value or "")
    except Exception:
        return ""


def default_checkpoint_dir(checkpoint_dir: Optional[str] = None) -> Path:
    if checkpoint_dir:
        return Path(checkpoint_dir).expanduser().resolve()
    return (_repo_root() / "checkpoints").resolve()


def task_checkpoint_root(*, checkpoint_dir: Optional[str], task_name: str) -> Path:
    return default_checkpoint_dir(checkpoint_dir) / _safe_name(task_name, default="task")


def build_run_id(
    *,
    model_name: str,
    language: str,
    run_label: Optional[str] = None,
) -> str:
    label = _safe_name(run_label or f"{model_name}_{language}", default="run")
    return f"{_utc_now_compact()}_{label}_{uuid.uuid4().hex[:8]}"


def _serialize_solution(solution: Any) -> dict[str, Any]:
    if isinstance(solution, Solution):
        return solution.to_dict()
    if hasattr(solution, "to_dict"):
        try:
            return solution.to_dict()
        except Exception:
            pass
    if hasattr(solution, "__dict__"):
        return dict(solution.__dict__)
    return {"solution": _stringify(solution)}


def _serialize_eval_result(eval_result: Optional[EvalResult]) -> Optional[dict[str, Any]]:
    if eval_result is None:
        return None
    try:
        return eval_result.to_dict(include_log_excerpt=True, max_log_chars=4000)
    except Exception:
        if hasattr(eval_result, "__dict__"):
            return dict(eval_result.__dict__)
        return {"eval_result": _stringify(eval_result)}


def _write_cleaned_code(*, round_dir: Path, cleaned_code: Any, language: str) -> list[str]:
    written: list[str] = []
    clean_dir = round_dir / "clean"
    lang = str(language or "").strip().lower()
    if lang == "cuda" and isinstance(cleaned_code, dict):
        for filename, content in cleaned_code.items():
            p = clean_dir / str(filename)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_stringify(content), encoding="utf-8")
            written.append(str(p))
        return written

    p = clean_dir / ("main.py" if lang in ("triton", "python") else "main.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_stringify(cleaned_code), encoding="utf-8")
    written.append(str(p))
    return written


@dataclass(frozen=True)
class RoundCheckpointRef:
    task_name: str
    run_id: str
    round_num: int
    run_dir: Path
    round_dir: Path
    solution_path: Path
    metadata_path: Path
    world_model_path: Optional[Path] = None


class RoundCheckpointManager:
    def __init__(
        self,
        *,
        checkpoint_dir: Optional[str],
        task_name: str,
        model_name: str,
        language: str,
        target_gpu: str,
        run_id: Optional[str] = None,
        run_label: Optional[str] = None,
    ) -> None:
        self.task_name = str(task_name or "")
        self.model_name = str(model_name or "")
        self.language = str(language or "")
        self.target_gpu = str(target_gpu or "")
        self.checkpoint_root = task_checkpoint_root(
            checkpoint_dir=checkpoint_dir,
            task_name=self.task_name,
        )
        self.runs_dir = self.checkpoint_root / "runs"
        self.run_id = str(run_id or build_run_id(model_name=self.model_name, language=self.language, run_label=run_label))
        self.run_dir = self.runs_dir / self.run_id
        self.rounds_dir = self.run_dir / "rounds"
        self.run_manifest_path = self.run_dir / "run.json"
        self.latest_run_path = self.checkpoint_root / "latest_run.json"
        self.created_at = _utc_now_iso()
        self._ensure_run_manifest(run_label=run_label)

    def _ensure_run_manifest(self, *, run_label: Optional[str]) -> None:
        if self.run_manifest_path.exists():
            return
        manifest = {
            "task_name": self.task_name,
            "run_id": self.run_id,
            "run_label": str(run_label or ""),
            "model_name": self.model_name,
            "language": self.language,
            "target_gpu": self.target_gpu,
            "created_at_utc": self.created_at,
            "updated_at_utc": self.created_at,
            "latest_round": 0,
            "best_round": None,
            "best_score": None,
            "best_solution_name": None,
            "rounds": [],
        }
        _json_dump(self.run_manifest_path, manifest)
        _json_dump(
            self.latest_run_path,
            {
                "task_name": self.task_name,
                "run_id": self.run_id,
                "updated_at_utc": self.created_at,
                "run_manifest_path": str(self.run_manifest_path),
            },
        )

    def save_round(
        self,
        *,
        round_num: int,
        solution: Any,
        cleaned_code: Any,
        raw_code: Any,
        eval_result: Optional[EvalResult],
        best_solution: Optional[Any],
        best_eval: Optional[EvalResult],
        best_score: Optional[float],
        prompt_text: Optional[str] = None,
        trace_logs: Optional[str] = None,
        world_model_json: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> RoundCheckpointRef:
        rn = int(round_num)
        round_dir = self.rounds_dir / f"r{rn:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        solution_path = round_dir / "solution.json"
        raw_code_path = round_dir / "raw_code.txt"
        eval_path = round_dir / "eval_result.json"
        metadata_path = round_dir / "metadata.json"
        prompt_path = round_dir / "prompt.txt"
        trace_logs_path = round_dir / "trace_logs.txt"
        world_model_path = round_dir / "world_model.json"

        solution_dict = _serialize_solution(solution)
        _json_dump(solution_path, solution_dict)
        raw_code_path.write_text(_stringify(raw_code), encoding="utf-8")
        written_clean_files = _write_cleaned_code(
            round_dir=round_dir,
            cleaned_code=cleaned_code,
            language=self.language,
        )

        eval_dict = _serialize_eval_result(eval_result)
        if eval_dict is not None:
            _json_dump(eval_path, eval_dict)

        if prompt_text is not None:
            prompt_path.write_text(_stringify(prompt_text), encoding="utf-8")
        if trace_logs is not None:
            trace_logs_path.write_text(_stringify(trace_logs), encoding="utf-8")
        if isinstance(world_model_json, str) and world_model_json.strip():
            world_model_path.write_text(world_model_json, encoding="utf-8")
            world_model_ref: Optional[Path] = world_model_path
        else:
            world_model_ref = None

        is_passed = bool(getattr(eval_result, "is_passed", lambda: False)()) if eval_result is not None else False
        score_name = None
        try:
            score_name = (
                eval_result.metrics.get("score_name")
                if eval_result is not None and isinstance(getattr(eval_result, "metrics", None), dict)
                else None
            )
        except Exception:
            score_name = None

        round_meta = {
            "task_name": self.task_name,
            "run_id": self.run_id,
            "round_num": rn,
            "saved_at_utc": _utc_now_iso(),
            "model_name": self.model_name,
            "language": self.language,
            "target_gpu": self.target_gpu,
            "solution_name": _stringify(getattr(solution, "name", "")).strip(),
            "solution_hash": (
                _stringify(solution.hash()).strip()
                if hasattr(solution, "hash")
                else None
            ),
            "is_passed": is_passed,
            "score": (float(best_score) if isinstance(best_score, (int, float)) else None)
            if best_solution is solution
            else (float(getattr(eval_result, "score", lambda: -1.0)()) if eval_result is not None and is_passed else None),
            "score_name": _stringify(score_name).strip() or None,
            "solution_path": str(solution_path),
            "eval_result_path": str(eval_path) if eval_dict is not None else None,
            "raw_code_path": str(raw_code_path),
            "clean_paths": written_clean_files,
            "prompt_path": str(prompt_path) if prompt_text is not None else None,
            "trace_logs_path": str(trace_logs_path) if trace_logs is not None else None,
            "world_model_path": str(world_model_ref) if world_model_ref is not None else None,
            "resume": {
                "continue_from_solution": str(solution_path),
                "continue_from_world_model": (str(world_model_ref) if world_model_ref is not None else None),
            },
            "best_so_far": {
                "solution_name": _stringify(getattr(best_solution, "name", "")).strip() or None,
                "score": (float(best_score) if isinstance(best_score, (int, float)) else None),
                "eval_result": _serialize_eval_result(best_eval),
            },
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            round_meta["extra"] = extra_metadata
        _json_dump(metadata_path, round_meta)

        self._update_run_manifest(
            round_num=rn,
            solution_name=round_meta["solution_name"],
            solution_path=solution_path,
            world_model_path=world_model_ref,
            eval_result=eval_result,
            best_solution=best_solution,
            best_score=best_score,
            metadata_path=metadata_path,
        )

        return RoundCheckpointRef(
            task_name=self.task_name,
            run_id=self.run_id,
            round_num=rn,
            run_dir=self.run_dir,
            round_dir=round_dir,
            solution_path=solution_path,
            metadata_path=metadata_path,
            world_model_path=world_model_ref,
        )

    def _update_run_manifest(
        self,
        *,
        round_num: int,
        solution_name: str,
        solution_path: Path,
        world_model_path: Optional[Path],
        eval_result: Optional[EvalResult],
        best_solution: Optional[Any],
        best_score: Optional[float],
        metadata_path: Path,
    ) -> None:
        manifest = _json_load(self.run_manifest_path) if self.run_manifest_path.exists() else {}
        rounds = manifest.get("rounds") if isinstance(manifest.get("rounds"), list) else []

        round_entry = {
            "round_num": int(round_num),
            "saved_at_utc": _utc_now_iso(),
            "solution_name": solution_name,
            "solution_path": str(solution_path),
            "metadata_path": str(metadata_path),
            "world_model_path": (str(world_model_path) if world_model_path is not None else None),
            "status": (_stringify(getattr(eval_result, "status", "")).strip() or None),
            "is_passed": (bool(getattr(eval_result, "is_passed", lambda: False)()) if eval_result is not None else False),
            "score": (float(getattr(eval_result, "score", lambda: -1.0)()) if eval_result is not None else None),
        }

        kept: list[dict[str, Any]] = []
        for item in rounds:
            if not isinstance(item, dict):
                continue
            if int(item.get("round_num", -1)) == int(round_num):
                continue
            kept.append(item)
        kept.append(round_entry)
        kept.sort(key=lambda item: int(item.get("round_num", 0)))

        manifest.update(
            {
                "task_name": self.task_name,
                "run_id": self.run_id,
                "model_name": self.model_name,
                "language": self.language,
                "target_gpu": self.target_gpu,
                "updated_at_utc": _utc_now_iso(),
                "latest_round": int(round_num),
                "latest_solution_name": solution_name,
                "latest_solution_path": str(solution_path),
                "latest_world_model_path": (str(world_model_path) if world_model_path is not None else None),
                "rounds": kept,
            }
        )

        best_name = _stringify(getattr(best_solution, "name", "")).strip() or None
        manifest["best_solution_name"] = best_name
        manifest["best_score"] = float(best_score) if isinstance(best_score, (int, float)) else None
        if best_name:
            best_round = None
            for item in kept:
                if str(item.get("solution_name", "") or "") == best_name:
                    best_round = int(item.get("round_num", 0))
            manifest["best_round"] = best_round

        _json_dump(self.run_manifest_path, manifest)
        _json_dump(
            self.latest_run_path,
            {
                "task_name": self.task_name,
                "run_id": self.run_id,
                "updated_at_utc": manifest.get("updated_at_utc"),
                "run_manifest_path": str(self.run_manifest_path),
                "latest_round": manifest.get("latest_round"),
            },
        )


def resolve_round_checkpoint(
    *,
    checkpoint_dir: Optional[str],
    task_name: str,
    run_id: str,
    round_num: Optional[int] = None,
) -> RoundCheckpointRef:
    task_root = task_checkpoint_root(checkpoint_dir=checkpoint_dir, task_name=task_name)
    run_dir = task_root / "runs" / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Checkpoint run not found: {run_dir}")

    manifest_path = run_dir / "run.json"
    manifest = _json_load(manifest_path) if manifest_path.exists() else {}
    resolved_round = int(round_num) if round_num is not None else int(manifest.get("latest_round") or 0)
    if resolved_round <= 0:
        round_dirs = sorted((run_dir / "rounds").glob("r*"))
        if not round_dirs:
            raise FileNotFoundError(f"No round checkpoints found under: {run_dir}")
        try:
            resolved_round = int(round_dirs[-1].name.lstrip("r"))
        except Exception as exc:
            raise FileNotFoundError(f"Could not infer latest round under: {run_dir}") from exc

    round_dir = run_dir / "rounds" / f"r{resolved_round:04d}"
    if not round_dir.exists():
        raise FileNotFoundError(f"Checkpoint round not found: {round_dir}")

    solution_path = round_dir / "solution.json"
    metadata_path = round_dir / "metadata.json"
    world_model_path = round_dir / "world_model.json"
    if not solution_path.exists():
        raise FileNotFoundError(f"Checkpoint solution missing: {solution_path}")

    return RoundCheckpointRef(
        task_name=str(task_name or ""),
        run_id=str(run_id or ""),
        round_num=int(resolved_round),
        run_dir=run_dir,
        round_dir=round_dir,
        solution_path=solution_path,
        metadata_path=metadata_path,
        world_model_path=(world_model_path if world_model_path.exists() else None),
    )


def list_checkpoint_runs(
    *,
    checkpoint_dir: Optional[str],
    task_name: str,
) -> list[dict[str, Any]]:
    task_root = task_checkpoint_root(checkpoint_dir=checkpoint_dir, task_name=task_name)
    runs_dir = task_root / "runs"
    if not runs_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_dir.glob("*/run.json")):
        try:
            data = _json_load(manifest_path)
        except Exception:
            continue
        out.append(data)
    out.sort(key=lambda item: str(item.get("created_at_utc", "")))
    return out
