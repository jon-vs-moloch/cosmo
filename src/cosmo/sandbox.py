"""Portable worker sandbox policy and path-boundary enforcement.

Cosmo is a pure-Python local harness. OS-level process containers, seccomp, or
network namespaces are not guaranteed across macOS/Linux developer machines.
This module therefore records that limitation explicitly and enforces a logical
fail-closed boundary:

- read-only accepted baseline under an approved ``repo_root``
- exactly one writable artifact slot (candidate workspace)
- no access to AtomicStore contents
- no shell/process execution, no network tools, no credentials
- bounded bytes / output / time metadata
- deterministic policy manifest + hash evidence

Claims of "container isolation" are intentionally absent; the durable policy
record is the source of truth for what was enforced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .atomic import (
    AtomicHarnessError,
    _assert_state_root_disjoint,
    _safe_relative_path,
)
from .reporting import canonical_hash


class SandboxViolation(AtomicHarnessError):
    """A sandbox boundary was crossed or a policy invariant failed."""


@dataclass(frozen=True)
class SandboxPolicy:
    """Explicit, portable sandbox boundary for one worker attempt.

    ``os_process_isolation`` is always ``False`` for this Python app: the
    harness cannot guarantee a portable process jail. Enforcement is logical
    (path checks, tool allowlists, durable logging) rather than OS-level.
    """

    schema_version: str = "cosmo.sandbox-policy.v1"
    repo_root: str = ""
    state_root: str = ""
    attempt_workspace: str = ""
    writable_artifact: str = ""
    baseline_mode: str = "read_only"
    max_writable_slots: int = 1
    allow_shell: bool = False
    allow_process_execution: bool = False
    allow_network: bool = False
    allow_credentials: bool = False
    allow_state_root_access: bool = False
    os_process_isolation: bool = False
    os_process_isolation_note: str = (
        "Portable OS-level process sandboxing is not guaranteed in this "
        "Python application; only the logical path/tool boundary is enforced."
    )
    max_read_bytes: int = 64 * 1024
    max_write_bytes: int = 1024 * 1024
    max_output_bytes: int = 1024 * 1024
    max_tool_calls: int = 256
    timeout_seconds: int | None = 300
    allowed_tools: tuple[str, ...] = (
        "list_tree",
        "read_artifact",
        "search_text",
        "inspect_python_symbols",
        "write_candidate",
    )
    forbidden_capabilities: tuple[str, ...] = (
        "shell",
        "subprocess",
        "network",
        "credentials",
        "secrets",
        "parent_integration_workspace",
        "arbitrary_file_handles",
        "product_tree_mutation",
        "atomic_store_read",
        "atomic_store_write",
    )
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.baseline_mode != "read_only":
            raise SandboxViolation(
                f"unsupported baseline_mode: {self.baseline_mode!r}"
            )
        if self.max_writable_slots != 1:
            raise SandboxViolation("exactly one writable artifact slot is required")
        if self.allow_shell or self.allow_process_execution:
            raise SandboxViolation("shell/process execution is forbidden")
        if self.allow_network:
            raise SandboxViolation("network access is forbidden for workers")
        if self.allow_credentials:
            raise SandboxViolation("credential/secret access is forbidden")
        if self.allow_state_root_access:
            raise SandboxViolation("AtomicStore/state-root access is forbidden")
        if self.os_process_isolation:
            raise SandboxViolation(
                "cannot claim os_process_isolation=True; portable process "
                "jailing is not available in this application"
            )
        if not self.repo_root or not self.writable_artifact:
            raise SandboxViolation(
                "SandboxPolicy requires repo_root and writable_artifact"
            )
        try:
            _assert_state_root_disjoint(
                Path(self.repo_root), Path(self.state_root)
            )
        except AtomicHarnessError as exc:
            raise SandboxViolation(str(exc)) from exc
        # Normalize path fields without mutating frozen fields via object.__setattr__.
        try:
            object.__setattr__(
                self, "writable_artifact", _safe_relative_path(self.writable_artifact)
            )
        except AtomicHarnessError as exc:
            raise SandboxViolation(str(exc)) from exc

    @property
    def policy_manifest(self) -> dict[str, Any]:
        """JSON-serializable policy record (deterministic field order via hash)."""
        body = {
            "schema_version": self.schema_version,
            "repo_root": str(Path(self.repo_root).resolve()) if self.repo_root else "",
            "state_root": str(Path(self.state_root).resolve()) if self.state_root else "",
            "attempt_workspace": (
                str(Path(self.attempt_workspace).resolve())
                if self.attempt_workspace
                else ""
            ),
            "writable_artifact": self.writable_artifact,
            "baseline_mode": self.baseline_mode,
            "max_writable_slots": self.max_writable_slots,
            "allow_shell": self.allow_shell,
            "allow_process_execution": self.allow_process_execution,
            "allow_network": self.allow_network,
            "allow_credentials": self.allow_credentials,
            "allow_state_root_access": self.allow_state_root_access,
            "os_process_isolation": self.os_process_isolation,
            "os_process_isolation_note": self.os_process_isolation_note,
            "max_read_bytes": self.max_read_bytes,
            "max_write_bytes": self.max_write_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_tool_calls": self.max_tool_calls,
            "timeout_seconds": self.timeout_seconds,
            "allowed_tools": list(self.allowed_tools),
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "extra": dict(self.extra),
        }
        body["policy_sha256"] = self.policy_sha256
        return body

    @property
    def policy_sha256(self) -> str:
        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "repo_root": str(Path(self.repo_root).resolve()) if self.repo_root else "",
                "state_root": str(Path(self.state_root).resolve()) if self.state_root else "",
                "attempt_workspace": (
                    str(Path(self.attempt_workspace).resolve())
                    if self.attempt_workspace
                    else ""
                ),
                "writable_artifact": self.writable_artifact,
                "baseline_mode": self.baseline_mode,
                "max_writable_slots": self.max_writable_slots,
                "allow_shell": self.allow_shell,
                "allow_process_execution": self.allow_process_execution,
                "allow_network": self.allow_network,
                "allow_credentials": self.allow_credentials,
                "allow_state_root_access": self.allow_state_root_access,
                "os_process_isolation": self.os_process_isolation,
                "os_process_isolation_note": self.os_process_isolation_note,
                "max_read_bytes": self.max_read_bytes,
                "max_write_bytes": self.max_write_bytes,
                "max_output_bytes": self.max_output_bytes,
                "max_tool_calls": self.max_tool_calls,
                "timeout_seconds": self.timeout_seconds,
                "allowed_tools": list(self.allowed_tools),
                "forbidden_capabilities": list(self.forbidden_capabilities),
                "extra": dict(self.extra),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return self.policy_manifest

    # ------------------------------------------------------------------
    # Boundary checks
    # ------------------------------------------------------------------

    def assert_tool_allowed(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise SandboxViolation(
                f"tool not permitted by sandbox policy: {tool_name!r}"
            )

    def safe_relative(self, relative_path: str) -> str:
        try:
            return _safe_relative_path(relative_path)
        except AtomicHarnessError as exc:
            raise SandboxViolation(str(exc)) from exc

    def resolve_under_root(self, root: Path, relative_path: str) -> Path:
        """Resolve a relative path under ``root``; refuse escapes and symlinks."""
        safe = self.safe_relative(relative_path)
        root = root.resolve()
        candidate = root / safe
        cursor = root
        for part in PurePosixPath(safe).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise SandboxViolation(
                    f"path contains a symlink which is not permitted: {safe}"
                )
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as exc:
            raise SandboxViolation(f"cannot resolve path: {safe}") from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise SandboxViolation(
                f"path escapes approved root: {relative_path!r}"
            ) from exc
        return resolved

    def assert_not_state_root(self, path: Path) -> None:
        if not self.state_root:
            return
        state = Path(self.state_root).resolve()
        try:
            path.resolve(strict=False).relative_to(state)
        except ValueError:
            return
        raise SandboxViolation(
            "access to AtomicStore/state root is forbidden by sandbox policy"
        )

    def assert_readable_path(self, relative_path: str) -> Path:
        """Approve a read under the product repo_root (not state, not escape)."""
        path = self.resolve_under_root(Path(self.repo_root), relative_path)
        self.assert_not_state_root(path)
        # Also refuse if the relative path itself enters state when nested.
        if self.state_root:
            try:
                state_rel = Path(self.state_root).resolve().relative_to(
                    Path(self.repo_root).resolve()
                )
            except ValueError:
                state_rel = None
            if state_rel is not None:
                prefix = state_rel.as_posix()
                safe = self.safe_relative(relative_path)
                if safe == prefix or safe.startswith(prefix + "/"):
                    raise SandboxViolation(
                        "access to AtomicStore/state root is forbidden by "
                        "sandbox policy"
                    )
        return path

    def assert_writable_candidate(self, relative_path: str) -> Path:
        """Approve write only for the single declared artifact in the attempt workspace.

        The attempt workspace is intentionally under AtomicStore (state root).
        Writes are allowed there for the one declared slot; all other state
        contents remain forbidden.
        """
        safe = self.safe_relative(relative_path)
        if safe != self.writable_artifact:
            raise SandboxViolation(
                f"write_candidate path {safe!r} is not the declared writable "
                f"artifact {self.writable_artifact!r}"
            )
        if not self.attempt_workspace:
            raise SandboxViolation("attempt workspace is not configured")
        workspace = Path(self.attempt_workspace).resolve()
        path = self.resolve_under_root(workspace, safe)
        try:
            path.resolve(strict=False).relative_to(workspace)
        except ValueError as exc:
            raise SandboxViolation(
                "writable candidate must stay inside the attempt workspace"
            ) from exc
        # Refuse product-tree mutation: product path for the same relative
        # artifact must not be the write target.
        product_target = Path(self.repo_root).resolve() / safe
        if path.resolve(strict=False) == product_target.resolve(strict=False):
            raise SandboxViolation(
                "write_candidate must not target the product tree directly"
            )
        return path

    def assert_write_size(self, content: str | bytes) -> None:
        if isinstance(content, str):
            size = len(content.encode("utf-8"))
        else:
            size = len(content)
        if size > self.max_write_bytes:
            raise SandboxViolation(
                f"candidate exceeds max_write_bytes ({self.max_write_bytes})"
            )

    def assert_output_size(self, content: str | bytes) -> None:
        """Enforce the provider-output budget before persisting a candidate."""
        if isinstance(content, str):
            size = len(content.encode("utf-8"))
        else:
            size = len(content)
        if size > self.max_output_bytes:
            raise SandboxViolation(
                f"model output exceeds max_output_bytes ({self.max_output_bytes})"
            )

    def assert_read_size(self, size_bytes: int) -> None:
        if size_bytes > self.max_read_bytes:
            raise SandboxViolation(
                f"read exceeds max_read_bytes ({self.max_read_bytes})"
            )

    def evidence_record(self) -> dict[str, Any]:
        """Deterministic hash evidence of the enforced boundary."""
        manifest = self.policy_manifest
        return {
            "schema_version": "cosmo.sandbox-evidence.v1",
            "policy_sha256": manifest["policy_sha256"],
            "policy": manifest,
            "evidence_sha256": canonical_hash(
                {"policy_sha256": manifest["policy_sha256"]}
            ),
        }


def build_worker_policy(
    *,
    repo_root: Path,
    state_root: Path,
    attempt_workspace: Path,
    writable_artifact: str,
    max_read_bytes: int = 64 * 1024,
    max_write_bytes: int = 1024 * 1024,
    timeout_seconds: int | None = 300,
    extra: Mapping[str, Any] | None = None,
) -> SandboxPolicy:
    """Construct the canonical fail-closed worker policy for one attempt."""
    return SandboxPolicy(
        repo_root=str(Path(repo_root).resolve()),
        state_root=str(Path(state_root).resolve()),
        attempt_workspace=str(Path(attempt_workspace).resolve()),
        writable_artifact=writable_artifact,
        max_read_bytes=max_read_bytes,
        max_write_bytes=max_write_bytes,
        timeout_seconds=timeout_seconds,
        extra=dict(extra or {}),
    )
