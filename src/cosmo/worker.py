"""Narrow worker tools and canonical WorkerExecutor.

Workers may only use structured read-only operations backed by
:class:`~cosmo.context.RepositoryInspector` plus a single-slot
``write_candidate``. Every tool call is scoped to the approved repo root /
attempt workspace and durably logged with argument and result hashes only —
never prompt or transcript bodies.

Inference itself remains the responsibility of :class:`~cosmo.atomic.AtomicHarness`;
this module does not re-implement run, validation, review, or promotion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic import (
    AtomicHarness,
    AtomicHarnessError,
    AtomicStore,
    _sha256_text,
    new_id,
    utc_now,
)
from .context import ContextError, RepositoryInspector
from .inference import InferenceClient
from .reporting import canonical_hash
from .sandbox import SandboxPolicy, SandboxViolation, build_worker_policy


class WorkerError(AtomicHarnessError):
    """Worker tool or execution boundary failure."""


class WorkerToolset:
    """Bounded tool surface for one attempt under a :class:`SandboxPolicy`."""

    def __init__(
        self,
        *,
        store: AtomicStore,
        policy: SandboxPolicy,
        attempt_id: str,
        inspector: RepositoryInspector | None = None,
    ):
        self.store = store
        self.policy = policy
        self.attempt_id = attempt_id
        self._assert_workspace_under_state_root()
        self.inspector = inspector or RepositoryInspector(
            Path(policy.repo_root),
            state_root=Path(policy.state_root) if policy.state_root else None,
            max_read_bytes=policy.max_read_bytes,
            max_map_entries=max(1, min(2000, policy.max_tool_calls * 8)),
            max_search_matches=50,
        )
        self._tool_call_count = 0
        self._candidate_written = False
        self._last_candidate_hash: str | None = None

    def _assert_workspace_under_state_root(self) -> None:
        """Refuse workspaces that are not under the AtomicStore state root."""
        if not self.policy.state_root or not self.policy.attempt_workspace:
            raise WorkerError(
                "WorkerToolset requires state_root and attempt_workspace on policy"
            )
        state = Path(self.policy.state_root).resolve()
        workspace = Path(self.policy.attempt_workspace).resolve()
        try:
            workspace.relative_to(state)
        except ValueError as exc:
            raise WorkerError(
                "attempt workspace must be under the AtomicStore state root; "
                f"workspace={workspace} state_root={state}"
            ) from exc

    def _coerce_positive_limit(self, name: str, value: Any) -> int:
        """Accept a positive int limit; raise WorkerError (never TypeError/ValueError).

        Booleans, floats, strings, and other non-integers are rejected so policy
        checks never leak built-in conversion exceptions to the caller.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkerError(
                f"{name} must be a positive integer (got {type(value).__name__})"
            )
        if value <= 0:
            raise WorkerError(f"{name} must be positive")
        return value

    def _assert_read_budget(
        self,
        *,
        max_bytes: int | None = None,
        max_entries: int | None = None,
        max_matches: int | None = None,
    ) -> None:
        """Reject over-policy or malformed limits before any filesystem read occurs.

        Non-integer / non-boolean-coercible values raise WorkerError rather than
        leaking ValueError or TypeError to the caller.
        """
        if max_bytes is not None:
            coerced = self._coerce_positive_limit("max_bytes", max_bytes)
            if coerced > self.policy.max_read_bytes:
                raise WorkerError(
                    f"requested max_bytes {coerced} exceeds sandbox policy "
                    f"max_read_bytes {self.policy.max_read_bytes}"
                )
        if max_entries is not None:
            coerced = self._coerce_positive_limit("max_entries", max_entries)
            # Bound list_tree fan-out relative to policy tool budget.
            hard_cap = max(1, self.policy.max_tool_calls * 8)
            if coerced > hard_cap:
                raise WorkerError(
                    f"requested max_entries {coerced} exceeds sandbox "
                    f"policy cap {hard_cap}"
                )
        if max_matches is not None:
            coerced = self._coerce_positive_limit("max_matches", max_matches)
            hard_cap = max(1, min(500, self.policy.max_tool_calls * 2))
            if coerced > hard_cap:
                raise WorkerError(
                    f"requested max_matches {coerced} exceeds sandbox "
                    f"policy cap {hard_cap}"
                )

    # ------------------------------------------------------------------
    # Durable hash-only tool logging
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_log_value(value: Any) -> Any:
        """Map a tool argument to a JSON-safe form for hashing and digests.

        Preserves JSON scalars, lists, and dicts (recursively). Unsupported
        values become a non-sensitive type marker only — never ``repr`` or
        raw content — so rejection logs cannot raise TypeError from
        :func:`canonical_hash` and cannot leak opaque object bodies.
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): WorkerToolset._sanitize_log_value(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [WorkerToolset._sanitize_log_value(item) for item in value]
        return {"type": type(value).__name__}

    def _log_tool(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: Any,
        status: str = "ok",
        error: str | None = None,
        count_call: bool = False,
    ) -> dict[str, Any]:
        # Calls are reserved in _guard before the operation executes.  Keeping
        # logging separate means a rejected call can be durably recorded
        # without performing the read/write and without incrementing the
        # budget a second time.
        if count_call:
            self._tool_call_count += 1
        # Never persist raw content or prompts — hashes only. Sanitize every
        # argument value first so non-JSON inputs (e.g. object()) never reach
        # canonical_hash or the durable digest.
        arg_payload: dict[str, Any] = {
            key: self._sanitize_log_value(value)
            for key, value in arguments.items()
        }
        # Sensitive string fields: length + hash exclusively (not raw text).
        if "content" in arguments and isinstance(arguments["content"], str):
            arg_payload["content"] = {
                "sha256": _sha256_text(arguments["content"]),
                "bytes": len(arguments["content"].encode("utf-8")),
            }
        if "query" in arguments and isinstance(arguments["query"], str):
            arg_payload["query"] = {
                "sha256": _sha256_text(arguments["query"]),
                "length": len(arguments["query"]),
            }
        if "pattern" in arguments and isinstance(arguments["pattern"], str):
            arg_payload["pattern"] = {
                "sha256": _sha256_text(arguments["pattern"]),
                "length": len(arguments["pattern"]),
            }
        result_hash = (
            canonical_hash(self._sanitize_log_value(result))
            if result is not None
            else None
        )
        event_id = new_id("TOOL")
        record = {
            "schema_version": "cosmo.worker-tool-event.v1",
            "tool_event_id": event_id,
            "attempt_id": self.attempt_id,
            "tool_name": tool_name,
            "arguments_hash": canonical_hash(arg_payload),
            "arguments_digest": arg_payload,
            "result_hash": result_hash,
            "status": status,
            "error": error,
            "policy_sha256": self.policy.policy_sha256,
            "sequence": self._tool_call_count,
            "created_at": utc_now(),
        }
        # Prefer dedicated kind when available; fall back to events only.
        if "worker_tool_events" in self.store.KINDS:
            self.store.write("worker_tool_events", event_id, record)
        self.store.append_event(
            "worker_tool_call",
            {
                "tool_event_id": event_id,
                "attempt_id": self.attempt_id,
                "tool_name": tool_name,
                "arguments_hash": record["arguments_hash"],
                "result_hash": result_hash,
                "status": status,
            },
        )
        return record

    def _guard(self, tool_name: str) -> None:
        if self._tool_call_count >= self.policy.max_tool_calls:
            message = (
                f"tool call budget exceeded ({self.policy.max_tool_calls})"
            )
            self._log_tool(
                tool_name=tool_name,
                arguments={},
                result=None,
                status="budget_exceeded",
                error=message,
            )
            raise WorkerError(message)
        # Reserve before any policy/path check or filesystem operation.  This
        # makes the budget an actual preflight boundary, not a post-operation
        # accounting check.
        self._tool_call_count += 1
        try:
            self.policy.assert_tool_allowed(tool_name)
        except SandboxViolation as exc:
            self._log_tool(
                tool_name=tool_name,
                arguments={},
                result=None,
                status="sandbox_violation",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Read-only tools
    # ------------------------------------------------------------------

    def list_tree(
        self, *, prefix: str = "", max_entries: int | None = None
    ) -> dict[str, Any]:
        self._guard("list_tree")
        # Safe args for rejection logs: no raw content (limits + prefix only).
        log_args = {"prefix": prefix, "max_entries": max_entries}
        try:
            self._assert_read_budget(max_entries=max_entries)
            if prefix:
                safe_prefix = prefix.rstrip("/")
                if safe_prefix:
                    self.policy.assert_readable_path(safe_prefix)
            result = self.inspector.list_tree(prefix=prefix, max_entries=max_entries)
        except WorkerError as exc:
            self._log_tool(
                tool_name="list_tree",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise
        except (ContextError, SandboxViolation, AtomicHarnessError) as exc:
            self._log_tool(
                tool_name="list_tree",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc
        self._log_tool(
            tool_name="list_tree",
            arguments=log_args,
            result={
                "entry_count": len(result.get("entries") or []),
                "result_sha256": result.get("result_sha256"),
            },
        )
        return result

    def read_artifact(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        self._guard("read_artifact")
        # Path + numeric limits only — never raw file content in rejection logs.
        log_args = {
            "path": path,
            "max_bytes": max_bytes,
            "start_line": start_line,
            "end_line": end_line,
        }
        try:
            # Policy limits must be enforced before any content is loaded.
            self._assert_read_budget(max_bytes=max_bytes)
            self.policy.assert_readable_path(path)
            effective_max = (
                self.policy.max_read_bytes
                if max_bytes is None
                else self._coerce_positive_limit("max_bytes", max_bytes)
            )
            result = self.inspector.read_artifact(
                path,
                max_bytes=effective_max,
                start_line=start_line,
                end_line=end_line,
            )
        except WorkerError as exc:
            self._log_tool(
                tool_name="read_artifact",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise
        except (ContextError, SandboxViolation, AtomicHarnessError) as exc:
            self._log_tool(
                tool_name="read_artifact",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc
        excerpt = result.get("excerpt") or ""
        if isinstance(excerpt, str):
            try:
                self.policy.assert_read_size(len(excerpt.encode("utf-8")))
            except SandboxViolation as exc:
                self._log_tool(
                    tool_name="read_artifact",
                    arguments=log_args,
                    result=None,
                    status="sandbox_violation",
                    error=str(exc),
                )
                raise WorkerError(str(exc)) from exc
        digest = {
            "path": result.get("path") or path,
            "sha256": result.get("sha256"),
            "content_sha256": result.get("content_sha256"),
            "size_bytes": result.get("size_bytes"),
            "truncated": result.get("truncated"),
        }
        self._log_tool(
            tool_name="read_artifact",
            arguments=log_args,
            result=digest,
        )
        return result

    def search_text(
        self,
        query: str,
        *,
        prefix: str = "",
        max_matches: int | None = None,
    ) -> dict[str, Any]:
        self._guard("search_text")
        # _log_tool hashes query; never persist raw search content on reject.
        log_args = {"query": query, "prefix": prefix, "max_matches": max_matches}
        try:
            self._assert_read_budget(max_matches=max_matches)
            if prefix:
                safe_prefix = prefix.rstrip("/")
                if safe_prefix:
                    self.policy.assert_readable_path(safe_prefix)
            result = self.inspector.search_text(
                query, prefix=prefix, max_matches=max_matches
            )
        except WorkerError as exc:
            self._log_tool(
                tool_name="search_text",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise
        except (ContextError, SandboxViolation, AtomicHarnessError) as exc:
            self._log_tool(
                tool_name="search_text",
                arguments=log_args,
                result=None,
                status="error",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc
        matches = result.get("matches") or []
        digest = {
            "match_count": len(matches),
            "truncated": result.get("truncated"),
            "result_sha256": result.get("result_sha256"),
            "paths": sorted(
                {
                    str(item.get("path"))
                    for item in matches
                    if isinstance(item, Mapping) and item.get("path")
                }
            )[:50],
        }
        self._log_tool(
            tool_name="search_text",
            arguments={"query": query, "prefix": prefix},
            result=digest,
        )
        return result

    def inspect_python_symbols(self, path: str) -> dict[str, Any]:
        self._guard("inspect_python_symbols")
        try:
            self.policy.assert_readable_path(path)
            result = self.inspector.inspect_python_symbols(path)
        except (ContextError, SandboxViolation, AtomicHarnessError) as exc:
            self._log_tool(
                tool_name="inspect_python_symbols",
                arguments={"path": path},
                result=None,
                status="error",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc
        symbols = result.get("symbols") or []
        digest = {
            "path": path,
            "symbol_count": len(symbols),
            "symbols_hash": canonical_hash(list(symbols)),
            "skipped": result.get("skipped"),
        }
        self._log_tool(
            tool_name="inspect_python_symbols",
            arguments={"path": path},
            result=digest,
        )
        return result

    # ------------------------------------------------------------------
    # Single writable slot
    # ------------------------------------------------------------------

    def write_candidate(self, path: str, content: str) -> dict[str, Any]:
        self._guard("write_candidate")
        if not isinstance(content, str):
            raise WorkerError("write_candidate content must be a string")
        try:
            target = self.policy.assert_writable_candidate(path)
            self.policy.assert_write_size(content)
        except SandboxViolation as exc:
            self._log_tool(
                tool_name="write_candidate",
                arguments={"path": path, "content": content},
                result=None,
                status="sandbox_violation",
                error=str(exc),
            )
            raise WorkerError(str(exc)) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        content_hash = _sha256_text(content)
        self._candidate_written = True
        self._last_candidate_hash = content_hash
        result = {
            "path": self.policy.writable_artifact,
            "sha256": content_hash,
            "bytes": len(content.encode("utf-8")),
            "workspace": str(Path(self.policy.attempt_workspace).resolve()),
        }
        self._log_tool(
            tool_name="write_candidate",
            arguments={"path": path, "content": content},
            result=result,
        )
        return result

    def tool_event_count(self) -> int:
        return self._tool_call_count


class WorkerExecutor:
    """Canonical worker executor: tools + AtomicHarness inference path.

    ``execute`` routes model work through :meth:`AtomicHarness.run_attempt`
    so validation, review, and promotion gates stay singular. Optional
    pre-attempt tool exploration is available via :meth:`open_toolset`.
    """

    def __init__(self, store: AtomicStore):
        self.store = store
        self.harness = AtomicHarness(store)

    def build_policy_for_attempt(self, attempt: Mapping[str, Any]) -> SandboxPolicy:
        return build_worker_policy(
            repo_root=Path(str(attempt["repo_root"])),
            state_root=self.store.root,
            attempt_workspace=Path(str(attempt["workspace"])),
            writable_artifact=str(attempt["writable_artifact"]),
            extra={"attempt_id": attempt.get("attempt_id")},
        )

    def open_toolset(
        self,
        *,
        attempt_id: str | None = None,
        repo_root: Path,
        workspace: Path,
        writable_artifact: str,
        attempt_id_for_log: str = "PENDING",
    ) -> WorkerToolset:
        """Open a toolset for an attempt workspace (or pre-attempt exploration)."""
        policy = build_worker_policy(
            repo_root=repo_root,
            state_root=self.store.root,
            attempt_workspace=workspace,
            writable_artifact=writable_artifact,
        )
        # Persist policy evidence once per open.
        evidence = policy.evidence_record()
        self.store.append_event(
            "sandbox_policy_bound",
            {
                "attempt_id": attempt_id or attempt_id_for_log,
                "policy_sha256": evidence["policy_sha256"],
                "os_process_isolation": False,
            },
        )
        return WorkerToolset(
            store=self.store,
            policy=policy,
            attempt_id=attempt_id or attempt_id_for_log,
        )

    def execute(
        self,
        *,
        task_id: str,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        strategy_summary: str,
        reasoning: str = "off",
        temperature: float = 0.0,
        worker_max_output_tokens: int = 2048,
        review_max_output_tokens: int = 2048,
        retry_contract: dict[str, Any] | None = None,
        read_only_paths: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Run one attempt through AtomicHarness (no parallel promotion path)."""
        result = self.harness.run_attempt(
            task_id=task_id,
            repo_root=Path(repo_root),
            client=client,
            model=model,
            reasoning=reasoning,
            temperature=temperature,
            worker_max_output_tokens=worker_max_output_tokens,
            review_max_output_tokens=review_max_output_tokens,
            strategy_summary=strategy_summary,
            retry_contract=retry_contract,
            read_only_paths=read_only_paths,
        )
        # Attach policy evidence for the completed attempt workspace.
        attempt = self.harness._read_attempt(str(result["attempt_id"]))
        policy = self.build_policy_for_attempt(attempt)
        result["sandbox_policy_sha256"] = policy.policy_sha256
        result["sandbox_os_process_isolation"] = False
        return result

    # Spec alias
    def execute_attempt(self, **kwargs: Any) -> dict[str, Any]:
        return self.execute(**kwargs)
