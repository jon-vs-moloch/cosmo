from __future__ import annotations

import hashlib
import plistlib
import subprocess
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_tree_hash(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def model_artifact_hash(model_dir: Path) -> str:
    files = [
        path
        for path in model_dir.rglob("*")
        if path.is_file() and not path.name.startswith("chat_template.jinja")
    ]
    if not files:
        raise ValueError(f"Model directory contains no artifact files: {model_dir}")
    return content_tree_hash(model_dir, files)


def git_state(repo: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {"commit": None, "dirty": True}
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


def lmstudio_app_version() -> dict[str, str] | None:
    candidates = [
        Path("/Applications/LM Studio.app/Contents/Info.plist"),
        Path.home() / "Applications/LM Studio.app/Contents/Info.plist",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("rb") as stream:
            value = plistlib.load(stream)
        return {
            "short_version": str(value.get("CFBundleShortVersionString") or ""),
            "build_version": str(value.get("CFBundleVersion") or ""),
            "bundle_path": str(path.parent.parent.parent),
        }
    return None


def run_content_hash(run_dir: Path) -> str:
    files = [path for path in run_dir.rglob("*") if path.is_file()]
    return content_tree_hash(run_dir, files)
