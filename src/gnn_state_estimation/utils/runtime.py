"""Runtime, hashing, and manifest helpers."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from .io import dump_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


def _import_torch():
    """Import torch only for runtime operations that require it.

    Lightweight helpers in this module, such as ``utc_now_iso`` and hashing
    utilities, are used by public-data snapshot tooling. Keeping PyTorch out of
    module import time lets those tools and their parsing tests run in minimal
    environments without pretending that model/evaluation code can run without
    the declared heavy dependency.
    """
    import torch

    return torch


def resolve_device(device_arg: str | None = None) -> torch.device:
    """Resolve and validate the requested device."""
    torch = _import_torch()
    requested = (device_arg or "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested} was requested, but CUDA is unavailable.")
        return torch.device(requested)
    raise ValueError(f"Unsupported device specifier: {device_arg!r}")


def _safe_run_git(args: list[str], cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    text = out.strip()
    return text or None


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_snapshot_files(repo_root: Path) -> Iterable[Path]:
    """Yield canonical source files for provenance hashing.

    This repo may be shared outside Git, so manifests also record a deterministic
    source snapshot hash over the main code, config, tests, and manuscript inputs.
    Generated results are intentionally excluded.
    """
    candidates = [
        repo_root / "requirements.txt",
        repo_root / "configs",
        repo_root / "scripts",
        repo_root / "src",
        repo_root / "tests",
        repo_root / "paper" / "main.tex",
        repo_root / "paper" / "references.bib",
    ]
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            resolved = child.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def sha256_source_snapshot(repo_root: str | Path) -> str | None:
    root = Path(repo_root)
    if not root.exists():
        return None
    h = hashlib.sha256()
    saw_file = False
    for path in _iter_snapshot_files(root):
        rel = path.relative_to(root.resolve()).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("utf-8"))
        h.update(b"\0")
        saw_file = True
    return h.hexdigest() if saw_file else None


def format_requirements_lock(requirements_path: str | Path) -> dict[str, Any]:
    path = Path(requirements_path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_text(text),
        "lines": [line.strip() for line in text.splitlines() if line.strip()],
    }


def collect_env_info(device: torch.device, requirements_path: str | Path = "requirements.txt") -> dict[str, Any]:
    torch = _import_torch()
    cuda_available = torch.cuda.is_available()
    gpu_index = 0 if device.type == "cuda" else None
    gpu_props = None
    if gpu_index is not None:
        props = torch.cuda.get_device_properties(gpu_index)
        gpu_props = {
            "name": torch.cuda.get_device_name(gpu_index),
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
            "major": int(props.major),
            "minor": int(props.minor),
        }
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "selected_device": str(device),
        "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu": gpu_props,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requirements": format_requirements_lock(requirements_path),
    }


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def duration_metadata(start_perf_counter: float, started_at_utc: str | None = None) -> dict[str, Any]:
    ended_at = utc_now_iso()
    return {
        "started_at_utc": started_at_utc or ended_at,
        "ended_at_utc": ended_at,
        "duration_sec": float(max(time.perf_counter() - start_perf_counter, 0.0)),
    }


def write_env_report(path: str | Path, device: torch.device, requirements_path: str | Path = "requirements.txt") -> dict[str, Any]:
    payload = collect_env_info(device=device, requirements_path=requirements_path)
    dump_json(payload, path)
    return payload


def build_run_manifest(
    *,
    command: list[str],
    config_text: str,
    config_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    seed: int,
    dataset_paths: dict[str, str | Path] | None = None,
    checkpoint_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_paths = dataset_paths or {}
    env_info = collect_env_info(device=device)
    repo_root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    git_commit = _safe_run_git(["rev-parse", "HEAD"], cwd=repo_root_path)
    git_branch = _safe_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root_path)
    source_snapshot_sha256 = sha256_source_snapshot(repo_root_path)
    manifest = {
        "command": command,
        "seed": int(seed),
        "config_path": str(config_path),
        "config_sha256": sha256_text(config_text),
        "device": env_info["selected_device"],
        "env": env_info,
        "datasets": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in dataset_paths.items() if Path(path).exists()
        },
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path is not None and Path(checkpoint_path).exists() else None,
        "repo_root": str(repo_root_path),
        "source_snapshot_sha256": source_snapshot_sha256,
        "vcs": {
            "backend": "git",
            "available": git_commit is not None,
            "commit": git_commit,
            "branch": git_branch,
        },
        "git_commit": git_commit,
        "git_branch": git_branch,
    }
    if timing:
        manifest["timing"] = timing
    if extra:
        manifest["extra"] = extra
    dump_json(manifest, output_path)
    return manifest
