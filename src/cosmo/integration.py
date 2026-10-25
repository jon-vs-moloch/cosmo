"""Parent IntegrationService: compose node-approved children, never product edits.

The Integrator applies selected content-addressed leaf Artifact versions into an
isolated workspace under the AtomicStore state root. Product files remain
untouched until an explicit promote operation after integration validation and
independent PASS review. No ad-hoc semantic edits are allowed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .atomic import (
    AtomicHarness,
    AtomicHarnessError,
    AtomicStaleError,
    AtomicStore,
    _artifact_target,
    _exclusive_lock,
    _product_revision,
    _product_snapshot,
    _read_text,
    _safe_record_identifier,
    _safe_relative_path,
    _sha256_bytes,
    _sha256_text,
    _write_text_atomic,
    deterministic_validate,
    new_id,
    parse_review,
    utc_now,
    REVIEW_SYSTEM,
    REVIEW_RESPONSE_FORMAT,
)
from .authority import (
    CallerProvenance,
    authenticate_model,
    require_model,
    require_operator,
)
from .inference import InferenceClient
from .reporting import canonical_hash


class IntegrationError(AtomicHarnessError):
    """Parent integration invariant or protocol failed."""


class IntegrationService:
    """Deterministic parent integration over AtomicStore + promotion lock."""

    def __init__(self, store: AtomicStore):
        self.store = store
        self.harness = AtomicHarness(store)

    # ------------------------------------------------------------------
    # Child approval / selection
    # ------------------------------------------------------------------

    def _read_plan(self, node_id: str) -> dict[str, Any]:
        return self.store.read("plan_nodes", _safe_record_identifier(node_id))

    def _direct_integration_children(
        self, parent: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve direct children without flattening nested parents.

        Nested parent children must already have an approved IntegrationRun
        (acceptance gate). Grandchildren are never silently composed as if that
        gate did not exist. Only direct leaf children participate in apply
        ordering for this parent.
        """
        child_ids = [str(item) for item in (parent.get("child_node_ids") or [])]
        leaves: list[dict[str, Any]] = []
        nested_parents: list[dict[str, Any]] = []
        for child_id in child_ids:
            child = self.store.read("plan_nodes", child_id)
            kind = child.get("kind")
            if kind == "leaf":
                leaves.append(child)
            elif kind == "parent":
                nested_parents.append(child)
                self._require_approved_nested_integration(child)
            else:
                raise IntegrationError(
                    f"unsupported plan child kind for integration: {kind!r} "
                    f"({child_id})"
                )
        return leaves, nested_parents

    def _require_approved_nested_integration(
        self, nested_parent: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Fail closed unless the nested parent already passed integration review."""
        node_id = str(nested_parent.get("node_id") or "")
        integration_id = nested_parent.get("integration_id")
        if not integration_id:
            raise IntegrationError(
                f"nested parent {node_id} has no approved IntegrationRun; "
                "compose and review that parent before ancestor integration "
                "(grandchildren are not flattened into the ancestor)"
            )
        try:
            run = self.store.read(
                "integration_runs", _safe_record_identifier(str(integration_id))
            )
        except AtomicHarnessError as exc:
            raise IntegrationError(
                f"nested parent {node_id} references missing IntegrationRun "
                f"{integration_id!r}"
            ) from exc
        if str(run.get("parent_node_id")) != node_id:
            raise IntegrationError(
                f"nested parent {node_id} integration_id does not match the "
                "stored IntegrationRun parent_node_id"
            )
        if not run.get("validation_passed"):
            raise IntegrationError(
                f"nested parent {node_id} IntegrationRun failed validation"
            )
        # Acceptance gate: review PASS and/or completed promotion.
        accepted = run.get("status") == "promoted" or (
            run.get("review_verdict") == "PASS"
            and run.get("status") in {"reviewed", "promoted"}
        )
        if not accepted and nested_parent.get("status") == "approved":
            accepted = run.get("review_verdict") == "PASS"
        if not accepted:
            raise IntegrationError(
                f"nested parent {node_id} has no approved IntegrationRun; "
                "compose and review that parent before ancestor integration "
                f"(status={run.get('status')!r}, "
                f"review_verdict={run.get('review_verdict')!r})"
            )
        return run

    def _leaf_children(self, parent: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Direct leaf children only (nested parents are gated, not flattened)."""
        leaves, _nested = self._direct_integration_children(parent)
        return leaves

    def _node_approved_artifact(
        self, leaf: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Return the latest promotable (validation+review PASS) artifact for a leaf.

        Node-approval does **not** require product promotion. Historical
        approvals remain on their attempts; we pick the newest reviewed PASS.
        """
        task_id = str(leaf.get("task_id") or "")
        if not task_id:
            return None
        attempts = [
            item
            for item in self.store.list_records("attempts")
            if item.get("task_id") == task_id
        ]
        # Prefer attempt_states for latest status.
        ranked: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for base in attempts:
            attempt_id = str(base["attempt_id"])
            state_path = self.store.path("attempt_states", attempt_id)
            attempt = (
                {**base, **self.store.read("attempt_states", attempt_id)}
                if state_path.is_file()
                else base
            )
            if not self._attempt_matches_current_spec(attempt, leaf):
                continue
            if not attempt.get("artifact_version_id"):
                continue
            if not attempt.get("validation_id") or not attempt.get("review_id"):
                continue
            try:
                validation = self.store.read(
                    "validations", str(attempt["validation_id"])
                )
                review = self.store.read("reviews", str(attempt["review_id"]))
            except AtomicHarnessError:
                continue
            if not validation.get("passed"):
                continue
            if review.get("verdict") != "PASS":
                continue
            artifact = self.store.read(
                "artifacts", str(attempt["artifact_version_id"])
            )
            ranked.append((attempt_id, attempt, artifact))
        if not ranked:
            return None
        # Newest by attempt_id (time-ordered id prefix).
        ranked.sort(key=lambda item: item[0])
        _aid, attempt, artifact = ranked[-1]
        return {
            "node_id": leaf.get("node_id"),
            "task_id": task_id,
            "attempt_id": attempt["attempt_id"],
            "artifact_version_id": artifact["version_id"],
            "logical_path": artifact["logical_path"],
            "content_hash": artifact["content_hash"],
            "content_path": artifact["content_path"],
            "validation_id": attempt["validation_id"],
            "review_id": attempt["review_id"],
            "baseline_tree_sha256": attempt.get("baseline_tree_sha256"),
            "dependencies": list(leaf.get("dependencies") or []),
        }

    def resolve_selected_children(
        self,
        parent_node_id: str,
        *,
        selected: Mapping[str, str] | None = None,
        require_all_leaves: bool = True,
    ) -> list[dict[str, Any]]:
        """Resolve explicit or latest node-approved child artifact versions.

        ``selected`` maps leaf ``node_id`` -> ``artifact_version_id``. When
        omitted, the latest node-approved version for each leaf is used.
        """
        parent = self._read_plan(parent_node_id)
        if parent.get("kind") != "parent":
            raise IntegrationError(
                f"assemble requires a parent plan node, got kind={parent.get('kind')!r}"
            )
        leaves, nested_parents = self._direct_integration_children(parent)
        if not leaves:
            if not nested_parents:
                raise IntegrationError("parent has no children to integrate")

        selected = dict(selected or {})
        resolved: list[dict[str, Any]] = []
        seen_paths: dict[str, str] = {}

        for leaf in leaves:
            node_id = str(leaf["node_id"])
            if selected and node_id not in selected:
                if require_all_leaves:
                    raise IntegrationError(
                        f"required leaf child not selected: {node_id}"
                    )
                continue
            if node_id in selected:
                version_id = _safe_record_identifier(str(selected[node_id]))
                artifact = self.store.read("artifacts", version_id)
                # Verify node-approval for this exact version.
                approved = self._node_approved_artifact(leaf)
                if approved is None or approved["artifact_version_id"] != version_id:
                    # Still allow if this specific version has PASS validation+review.
                    if not self._artifact_is_node_approved(version_id, leaf):
                        raise IntegrationError(
                            f"selected artifact {version_id} is not node-approved "
                            f"for leaf {node_id}"
                        )
                entry = {
                    "node_id": node_id,
                    "task_id": leaf.get("task_id"),
                    "attempt_id": self._attempt_for_artifact(version_id),
                    "artifact_version_id": version_id,
                    "logical_path": artifact["logical_path"],
                    "content_hash": artifact["content_hash"],
                    "content_path": artifact["content_path"],
                    "dependencies": list(leaf.get("dependencies") or []),
                    "composition_rank": 1,
                }
            else:
                approved = self._node_approved_artifact(leaf)
                if approved is None:
                    raise IntegrationError(
                        f"leaf child is not node-approved: {node_id}"
                    )
                entry = approved
                entry["composition_rank"] = 1

            path = _safe_relative_path(str(entry["logical_path"]))
            if path in seen_paths:
                raise IntegrationError(
                    f"duplicate logical path {path!r} from children "
                    f"{seen_paths[path]} and {entry['node_id']}"
                )
            # Writable path on the leaf plan must match the artifact.
            leaf_path = leaf.get("writable_artifact")
            if leaf_path and _safe_relative_path(str(leaf_path)) != path:
                raise IntegrationError(
                    f"artifact path {path!r} differs from leaf writable_artifact "
                    f"{leaf_path!r}"
                )
            entry["logical_path"] = path
            seen_paths[path] = str(entry["node_id"])
            resolved.append(entry)

        # A nested parent is an already-reviewed content-addressed result, not
        # a reason to silently flatten/re-run its grandchildren.  Overlay the
        # exact approved artifact versions recorded by that IntegrationRun so
        # recursive composition contributes bytes to the ancestor workspace.
        for nested_parent in nested_parents:
            run = self._require_approved_nested_integration(nested_parent)
            nested_id = str(nested_parent["node_id"])
            for item in run.get("applied_paths") or []:
                logical = _safe_relative_path(str(item["logical_path"]))
                if logical in seen_paths:
                    raise IntegrationError(
                        f"duplicate logical path {logical!r} from nested parent "
                        f"{nested_id} and {seen_paths[logical]}"
                    )
                version_id = _safe_record_identifier(
                    str(item["artifact_version_id"])
                )
                artifact = self.store.read("artifacts", version_id)
                content_path = Path(str(artifact.get("content_path") or ""))
                if not content_path.is_file():
                    raise IntegrationError(
                        f"nested integration artifact bytes are missing: {version_id}"
                    )
                content_hash = _sha256_bytes(content_path.read_bytes())
                if content_hash != artifact.get("content_hash") or content_hash != item.get("content_hash"):
                    raise IntegrationError(
                        f"nested integration artifact hash mismatch: {version_id}"
                    )
                entry = {
                    "node_id": f"{nested_id}:{item.get('node_id') or version_id}",
                    "source_parent_node_id": nested_id,
                    "task_id": None,
                    "attempt_id": item.get("attempt_id"),
                    "artifact_version_id": version_id,
                    "logical_path": logical,
                    "content_hash": content_hash,
                    "content_path": str(content_path),
                    # The nested run has already topologically ordered and
                    # reviewed its internal children; treat it as an atomic
                    # composition unit in this ancestor.
                    "dependencies": [],
                    "composition_rank": 0,
                }
                seen_paths[logical] = str(entry["node_id"])
                resolved.append(entry)

        return self._topological_order(resolved)

    def _artifact_is_node_approved(
        self, version_id: str, leaf: Mapping[str, Any]
    ) -> bool:
        for attempt in self.store.list_records("attempts"):
            if attempt.get("artifact_version_id") != version_id:
                # Check state projection.
                state_path = self.store.path(
                    "attempt_states", str(attempt["attempt_id"])
                )
                if state_path.is_file():
                    state = self.store.read(
                        "attempt_states", str(attempt["attempt_id"])
                    )
                    if state.get("artifact_version_id") != version_id:
                        continue
                    attempt = {**attempt, **state}
                else:
                    continue
            if attempt.get("task_id") != leaf.get("task_id"):
                continue
            if not self._attempt_matches_current_spec(attempt, leaf):
                continue
            if not attempt.get("validation_id") or not attempt.get("review_id"):
                continue
            try:
                validation = self.store.read(
                    "validations", str(attempt["validation_id"])
                )
                review = self.store.read("reviews", str(attempt["review_id"]))
            except AtomicHarnessError:
                continue
            if validation.get("passed") and review.get("verdict") == "PASS":
                return True
        return False

    def _attempt_matches_current_spec(
        self, attempt: Mapping[str, Any], leaf: Mapping[str, Any]
    ) -> bool:
        """Reject reviewed bytes from a superseded TaskSpec or stale attempt."""
        if attempt.get("stale") or attempt.get("stale_reason"):
            return False
        task_id = str(leaf.get("task_id") or attempt.get("task_id") or "")
        if not task_id:
            return False
        try:
            task = self.store.read("tasks", task_id)
            spec = self.store.read("task_specs", str(task.get("current_spec_id")))
        except AtomicHarnessError:
            return False
        if attempt.get("spec_id") != task.get("current_spec_id"):
            return False
        if int(attempt.get("spec_version") or spec.get("version") or 0) != int(
            task.get("current_spec_version") or spec.get("version") or 0
        ):
            return False
        if spec.get("superseded_by"):
            return False
        return True

    def _attempt_for_artifact(self, version_id: str) -> str | None:
        for attempt in self.store.list_records("attempts"):
            if attempt.get("artifact_version_id") == version_id:
                return str(attempt["attempt_id"])
            state_path = self.store.path("attempt_states", str(attempt["attempt_id"]))
            if state_path.is_file():
                state = self.store.read("attempt_states", str(attempt["attempt_id"]))
                if state.get("artifact_version_id") == version_id:
                    return str(attempt["attempt_id"])
        return None

    def _topological_order(
        self, children: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Order by plan-node dependencies; fall back to path order for ties."""
        by_node = {str(item["node_id"]): dict(item) for item in children}
        # Dependencies may reference plan node ids outside this selection;
        # only edges among selected nodes constrain order.
        indegree = {nid: 0 for nid in by_node}
        edges: dict[str, list[str]] = {nid: [] for nid in by_node}
        for nid, item in by_node.items():
            for dep in item.get("dependencies") or []:
                dep_id = str(dep)
                if dep_id in by_node and dep_id != nid:
                    edges[dep_id].append(nid)
                    indegree[nid] += 1
        ready = sorted(
            [nid for nid, deg in indegree.items() if deg == 0],
            key=lambda n: (
                int(by_node[n].get("composition_rank") or 1),
                str(by_node[n]["logical_path"]),
                n,
            ),
        )
        ordered: list[dict[str, Any]] = []
        while ready:
            nid = ready.pop(0)
            ordered.append(by_node[nid])
            for child in sorted(edges[nid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort(
                key=lambda n: (
                    int(by_node[n].get("composition_rank") or 1),
                    str(by_node[n]["logical_path"]),
                    n,
                )
            )
        if len(ordered) != len(by_node):
            raise IntegrationError(
                "child dependency graph contains a cycle among selected leaves"
            )
        return ordered

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------

    def assemble(
        self,
        parent_node_id: str,
        *,
        repo_root: Path,
        selected: Mapping[str, str] | None = None,
        require_all_leaves: bool = True,
        expected_baseline_tree_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Create/refresh an isolated integration workspace and apply children."""
        parent_node_id = _safe_record_identifier(parent_node_id)
        parent = self._read_plan(parent_node_id)
        if parent.get("kind") != "parent":
            raise IntegrationError("assemble requires a parent plan node")
        repo_root = Path(repo_root).resolve()
        if not repo_root.is_dir():
            raise IntegrationError(f"repo_root is not a directory: {repo_root}")

        baseline_tree = _product_revision(repo_root, excluded_root=self.store.root)
        if (
            expected_baseline_tree_sha256 is not None
            and expected_baseline_tree_sha256 != baseline_tree
        ):
            raise IntegrationError(
                "stale baseline: product tree sha differs from expected"
            )

        # Gate nested parents before resolving leaves (no silent flatten).
        parent_for_children = self._read_plan(parent_node_id)
        _leaves, nested_parents = self._direct_integration_children(parent_for_children)

        children = self.resolve_selected_children(
            parent_node_id,
            selected=selected,
            require_all_leaves=require_all_leaves,
        )

        integration_id = new_id("INTEG")
        workspace = self.store.root / "integration_workspaces" / integration_id
        if workspace.exists():
            raise IntegrationError(f"integration workspace already exists: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)

        # Seed workspace from a full isolated copy of the accepted baseline
        # (exclude .git and AtomicStore), then overlay selected child artifacts.
        baseline_snapshot = _product_snapshot(
            repo_root, excluded_root=self.store.root
        )
        self._seed_workspace_from_baseline(
            repo_root=repo_root,
            workspace=workspace,
            baseline_snapshot=baseline_snapshot,
        )
        applied_paths: list[dict[str, Any]] = []
        for child in children:
            path = str(child["logical_path"])
            # Refuse unsafe / state-root destinations.
            try:
                safe = _safe_relative_path(path)
            except AtomicHarnessError as exc:
                raise IntegrationError(str(exc)) from exc
            if safe.startswith(".cosmo/") or "/.cosmo/" in f"/{safe}/":
                raise IntegrationError(
                    f"refusing to apply into state-like path: {safe}"
                )
            content_path = Path(str(child["content_path"]))
            if not content_path.is_file():
                raise IntegrationError(
                    f"missing content-addressed artifact bytes: {content_path}"
                )
            raw = content_path.read_bytes()
            if _sha256_bytes(raw) != child["content_hash"]:
                raise IntegrationError(
                    f"artifact content hash mismatch for {child['artifact_version_id']}"
                )
            dest = workspace / safe
            # Symlink refusal: do not follow escapes.
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise IntegrationError(f"refusing symlink destination: {safe}")
            _write_text_atomic(dest, raw.decode("utf-8"))
            applied_paths.append(
                {
                    "logical_path": safe,
                    "artifact_version_id": child["artifact_version_id"],
                    "node_id": child["node_id"],
                    "content_hash": child["content_hash"],
                    "attempt_id": child.get("attempt_id"),
                }
            )

        workspace_snapshot = {
            path.relative_to(workspace).as_posix(): _sha256_bytes(path.read_bytes())
            for path in sorted(workspace.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }
        # Undeclared writes: any path whose content differs from the baseline
        # (or is new) must be in the declared apply set. Baseline files that
        # remain unchanged are not writes.
        declared = {item["logical_path"] for item in applied_paths}
        changed_paths = sorted(
            path
            for path, digest in workspace_snapshot.items()
            if baseline_snapshot.get(path) != digest
        )
        undeclared = [path for path in changed_paths if path not in declared]
        if undeclared:
            raise IntegrationError(
                f"undeclared writes in integration workspace: {undeclared}"
            )
        # Every declared path must actually differ or be present after apply.
        missing_declared = sorted(declared - set(workspace_snapshot))
        if missing_declared:
            raise IntegrationError(
                f"declared apply paths missing from workspace: {missing_declared}"
            )

        checks = [
            {
                "name": "all_required_children_node_approved",
                "result": "pass",
                "evidence": f"{len(children)} children",
            },
            {
                "name": "nested_parents_acceptance_gated",
                "result": "pass",
                "evidence": [
                    {
                        "node_id": item.get("node_id"),
                        "integration_id": item.get("integration_id"),
                        "status": item.get("status"),
                    }
                    for item in nested_parents
                ],
            },
            {
                "name": "no_duplicate_paths",
                "result": "pass",
                "evidence": f"{len(declared)} unique paths",
            },
            {
                "name": "no_undeclared_writes",
                "result": "pass",
                "evidence": {
                    "changed_path_count": len(changed_paths),
                    "baseline_path_count": len(baseline_snapshot),
                    "declared_path_count": len(declared),
                },
            },
            {
                "name": "content_addressed_apply",
                "result": "pass",
                "evidence": "each child hash matched content store",
            },
            {
                "name": "baseline_current",
                "result": "pass",
                "evidence": baseline_tree,
            },
            {
                "name": "dependency_order",
                "result": "pass",
                "evidence": [item["node_id"] for item in children],
            },
            {
                "name": "baseline_seeded",
                "result": "pass",
                "evidence": f"copied {len(baseline_snapshot)} baseline paths",
            },
        ]
        # Optional parent composition validators from the plan node.
        parent_validators = list(parent.get("validators") or [])
        for name in parent_validators:
            if name in {"non_empty", "utf8", "json", "python_syntax", "python"}:
                # Run against each applied artifact.
                for item in applied_paths:
                    content = (workspace / item["logical_path"]).read_text(
                        encoding="utf-8"
                    )
                    result = deterministic_validate(
                        content=content,
                        artifact_path=item["logical_path"],
                        validators=[name],
                    )
                    checks.append(
                        {
                            "name": f"parent_validator:{name}:{item['logical_path']}",
                            "result": "pass" if result["passed"] else "fail",
                            "evidence": result["checks"],
                        }
                    )
            else:
                checks.append(
                    {
                        "name": f"parent_validator:{name}",
                        # A declared validator without an implementation is
                        # not evidence of a passing composition.  Keep the
                        # record explicit and fail closed until a
                        # language-/contract-specific adapter is registered.
                        "result": "fail",
                        "evidence": (
                            "no deterministic adapter registered for parent "
                            f"validator {name!r}"
                        ),
                    }
                )

        passed = all(item["result"] == "pass" for item in checks)
        diff = {
            "applied": applied_paths,
            "baseline_tree_sha256": baseline_tree,
            "workspace_tree_sha256": canonical_hash(workspace_snapshot),
            "paths": sorted(declared),
            "changed_paths": changed_paths,
            "baseline_path_count": len(baseline_snapshot),
        }

        record = {
            "schema_version": "cosmo.integration-run.v1",
            "integration_id": integration_id,
            "parent_node_id": parent_node_id,
            "repo_root": str(repo_root),
            "workspace": str(workspace),
            "baseline_tree_sha256": baseline_tree,
            "baseline_snapshot_sha256": canonical_hash(baseline_snapshot),
            "selected_children": [
                {
                    "node_id": c["node_id"],
                    "task_id": c.get("task_id"),
                    "attempt_id": c.get("attempt_id"),
                    "artifact_version_id": c["artifact_version_id"],
                    "logical_path": c["logical_path"],
                    "content_hash": c["content_hash"],
                    "dependencies": list(c.get("dependencies") or []),
                }
                for c in children
            ],
            "nested_parents": [
                {
                    "node_id": item.get("node_id"),
                    "integration_id": item.get("integration_id"),
                    "status": item.get("status"),
                }
                for item in nested_parents
            ],
            "applied_paths": applied_paths,
            "diff": diff,
            "checks": checks,
            "validation_passed": passed,
            "status": "assembled" if passed else "failed",
            "review_id": None,
            "review_verdict": None,
            "promotable": False,
            "created_at": utc_now(),
        }
        self.store.write("integration_runs", integration_id, record)
        self.store.append_event(
            "integration_assembled",
            {
                "integration_id": integration_id,
                "parent_node_id": parent_node_id,
                "child_count": len(children),
                "validation_passed": passed,
                "workspace_tree_sha256": diff["workspace_tree_sha256"],
            },
        )
        return record

    def _seed_workspace_from_baseline(
        self,
        *,
        repo_root: Path,
        workspace: Path,
        baseline_snapshot: Mapping[str, str],
    ) -> None:
        """Copy every accepted product file into the integration workspace."""
        repo_root = Path(repo_root).resolve()
        for relative, expected_hash in baseline_snapshot.items():
            try:
                safe = _safe_relative_path(relative)
            except AtomicHarnessError as exc:
                raise IntegrationError(str(exc)) from exc
            source = repo_root / safe
            if not source.is_file() or source.is_symlink():
                raise IntegrationError(
                    f"baseline file missing or is symlink during seed: {safe}"
                )
            raw = source.read_bytes()
            if _sha256_bytes(raw) != expected_hash:
                raise IntegrationError(
                    f"baseline content changed during workspace seed: {safe}"
                )
            dest = workspace / safe
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise IntegrationError(f"refusing symlink destination: {safe}")
            dest.write_bytes(raw)

    # ------------------------------------------------------------------
    # Review projection
    # ------------------------------------------------------------------

    def project_for_review(self, integration_id: str) -> dict[str, Any]:
        """Build a review/approval projection without mutating product files."""
        integration_id = _safe_record_identifier(integration_id)
        run = self.store.read("integration_runs", integration_id)
        return {
            "schema_version": "cosmo.integration-review-projection.v1",
            "integration_id": integration_id,
            "parent_node_id": run["parent_node_id"],
            "status": run.get("status"),
            "validation_passed": run.get("validation_passed"),
            "selected_children": run.get("selected_children"),
            "applied_paths": run.get("applied_paths"),
            "diff": run.get("diff"),
            "checks": run.get("checks"),
            "baseline_tree_sha256": run.get("baseline_tree_sha256"),
            "repo_root": run.get("repo_root"),
            "workspace": run.get("workspace"),
        }

    def record_review(
        self,
        integration_id: str,
        *,
        verdict: str,
        requirement_results: Sequence[Mapping[str, Any]] | None = None,
        notes: str | None = None,
        independent_reviewer: str = "operator_integration_review",
        generation: Mapping[str, Any] | None = None,
        operator_caller: CallerProvenance | None = None,
        model_caller: CallerProvenance | None = None,
    ) -> dict[str, Any]:
        """Record an independent integration review (operator or model)."""
        operator_supplied = generation is None
        if operator_supplied:
            caller_provenance = require_operator(operator_caller).to_record()
        else:
            caller_provenance = require_model(model_caller).to_record()
        integration_id = _safe_record_identifier(integration_id)
        run = self.store.read("integration_runs", integration_id)
        if not run.get("validation_passed"):
            raise IntegrationError(
                "cannot review a failed integration assembly"
            )
        if verdict not in {"PASS", "REVISE", "REPLAN", "BLOCKED", "STALE", "ESCALATE"}:
            raise IntegrationError(f"unsupported integration review verdict: {verdict!r}")
        reqs = list(requirement_results or [])
        if verdict == "PASS" and not reqs:
            reqs = [
                {
                    "requirement_id": "composition",
                    "result": "PASS",
                    "evidence": ["integration deterministic checks passed"],
                }
            ]
        if verdict == "PASS" and any(
            item.get("result") != "PASS" for item in reqs
        ):
            raise IntegrationError(
                "PASS integration review cannot contain failed requirements"
            )
        review_id = new_id("INTEG-REVIEW")
        review = {
            "schema_version": "cosmo.integration-review.v1",
            "review_id": review_id,
            "integration_id": integration_id,
            "verdict": verdict,
            "requirement_results": reqs,
            "notes": notes,
            "independent_reviewer": independent_reviewer,
            "operator_supplied": operator_supplied,
            "caller_provenance": caller_provenance,
            "generation": dict(generation or {}) or None,
            "created_at": utc_now(),
        }
        if "integration_reviews" in self.store.KINDS:
            self.store.write("integration_reviews", review_id, review)
        else:
            self.store.write("reviews", review_id, review)

        run["review_id"] = review_id
        run["review_verdict"] = verdict
        run["status"] = "reviewed"
        run["promotable"] = bool(
            run.get("validation_passed") and verdict == "PASS"
        )
        run["reviewed_at"] = utc_now()
        self.store.replace("integration_runs", integration_id, run)
        self.store.append_event(
            "integration_reviewed",
            {
                "integration_id": integration_id,
                "review_id": review_id,
                "verdict": verdict,
                "promotable": run["promotable"],
            },
        )
        return {"review": review, "integration": run}

    def review_with_client(
        self,
        integration_id: str,
        *,
        client: InferenceClient,
        model: str,
        reasoning: str = "off",
        max_output_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Optional model-backed integration review through the local client only."""
        projection = self.project_for_review(integration_id)
        prompt = f"""INTEGRATION REVIEW PROJECTION
{json.dumps(projection, indent=2, sort_keys=True)}

Review the parent integration against composition requirements. Return the
required JSON review object. A PASS verdict requires every requirement PASS."""
        generation = client.chat(
            model=model,
            role="atomic_reviewer",
            system_prompt=REVIEW_SYSTEM,
            prompt=prompt,
            reasoning=reasoning,
            temperature=0,
            max_output_tokens=max_output_tokens,
            response_format=REVIEW_RESPONSE_FORMAT,
            operation_id=f"integration/{integration_id}/review",
        )
        parsed = parse_review(generation.content)
        return self.record_review(
            integration_id,
            verdict=str(parsed["verdict"]),
            requirement_results=parsed["requirement_results"],
            notes=str(parsed.get("recommended_next_action") or ""),
            independent_reviewer="atomic_reviewer",
            generation={
                "role": generation.role,
                "content_sha256": _sha256_text(generation.content),
                "request_hash": generation.request_hash,
                "reasoning_hash": generation.reasoning_hash,
                "stats": dict(generation.stats or {}),
                "latency_seconds": generation.latency_seconds,
                "model_instance_id": generation.model_instance_id,
            },
            model_caller=authenticate_model(
                f"lmstudio:{model}",
                authenticated_by="cosmo.integration-review",
            ),
        )

    # ------------------------------------------------------------------
    # Promote (explicit, after PASS)
    # ------------------------------------------------------------------

    def promote(
        self,
        integration_id: str,
        *,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        """Apply node-approved children to the product tree under the promotion lock.

        Product files are unchanged until this method succeeds. Uses the same
        exclusive promotion lock convention as AtomicHarness.promote.
        """
        integration_id = _safe_record_identifier(integration_id)
        run = self.store.read("integration_runs", integration_id)
        if not run.get("validation_passed"):
            raise IntegrationError(
                "integration promotion requires deterministic validation PASS"
            )
        if run.get("review_verdict") != "PASS":
            raise IntegrationError(
                "integration promotion requires independent review PASS"
            )
        if not run.get("promotable"):
            raise IntegrationError("integration run is not promotable")
        if run.get("status") == "promoted":
            existing = [
                item
                for item in self.store.list_records("promotions")
                if item.get("integration_id") == integration_id
            ]
            if existing:
                return existing[-1]

        root = Path(repo_root or run["repo_root"]).resolve()
        if root != Path(str(run["repo_root"])).resolve():
            raise IntegrationError(
                "promotion repo_root differs from integration baseline"
            )

        with _exclusive_lock(self.store.promotion_lock(root)):
            recovered = self._recover_prepared_intent(
                integration_id=integration_id, repo_root=root
            )
            if recovered is not None:
                return recovered
            return self._promote_locked(run, repo_root=root)

    def _recover_prepared_intent(
        self, *, integration_id: str, repo_root: Path
    ) -> dict[str, Any] | None:
        """Finalize an integration promotion whose writes landed before a crash.

        A prepared intent is committed only when the product tree already equals
        the intent's recorded post-write tree. Any other observed tree falls
        through to the normal gated path, which re-checks baseline staleness
        and fails closed on a torn or externally modified product.
        """
        intents = [
            item
            for item in self.store.list_records("integration_promotion_intents")
            if item.get("integration_id") == integration_id
            and item.get("status") == "prepared"
        ]
        if not intents:
            return None
        intent = intents[-1]
        if Path(str(intent.get("repo_root"))).resolve() != repo_root.resolve():
            raise IntegrationError(
                "promotion repo_root differs from the prepared "
                "integration intent"
            )
        current_tree = _product_revision(repo_root, excluded_root=self.store.root)
        if current_tree != intent.get("post_tree_sha256"):
            return None
        run = self.store.read("integration_runs", integration_id)
        if not (
            run.get("validation_passed")
            and run.get("review_verdict") == "PASS"
            and run.get("promotable")
        ):
            raise IntegrationError(
                "prepared integration intent exists but the run is no "
                "longer promotable"
            )
        existing = [
            item
            for item in self.store.list_records("promotions")
            if item.get("integration_id") == integration_id
        ]
        return self._complete_promotion(
            run,
            repo_root=repo_root,
            promotion=existing[-1] if existing else None,
            intent_id=str(intent["intent_id"]),
            recovered=True,
        )

    def _prepare_promotion_intent(
        self,
        run: Mapping[str, Any],
        *,
        staged_entries: Sequence[Mapping[str, Any]],
        repo_root: Path,
        post_tree_sha256: str,
    ) -> str:
        """Write or reuse the durable intent before any product write."""
        integration_id = str(run["integration_id"])
        prepared = [
            item
            for item in self.store.list_records("integration_promotion_intents")
            if item.get("integration_id") == integration_id
            and item.get("status") == "prepared"
        ]
        entries = [dict(item) for item in staged_entries]
        if prepared:
            intent = prepared[-1]
            intent_id = str(intent["intent_id"])
            if (
                intent.get("post_tree_sha256") != post_tree_sha256
                or Path(str(intent.get("repo_root"))).resolve()
                != repo_root.resolve()
                or [dict(item) for item in intent.get("applied_paths") or []]
                != entries
            ):
                raise IntegrationError(
                    "prepared integration promotion intent does not match "
                    "the staged candidate"
                )
            return intent_id
        intent_id = new_id("IPINTENT")
        self.store.write(
            "integration_promotion_intents",
            intent_id,
            {
                "schema_version": "cosmo.integration-promotion-intent.v1",
                "intent_id": intent_id,
                "integration_id": integration_id,
                "parent_node_id": run["parent_node_id"],
                "repo_root": str(repo_root.resolve()),
                "baseline_tree_sha256": run.get("baseline_tree_sha256"),
                "post_tree_sha256": post_tree_sha256,
                "applied_paths": entries,
                "status": "prepared",
                "created_at": utc_now(),
            },
        )
        self.store.append_event(
            "integration_promotion_prepared",
            {"intent_id": intent_id, "integration_id": integration_id},
        )
        return intent_id

    def _complete_promotion(
        self,
        run: Mapping[str, Any],
        *,
        repo_root: Path,
        promotion: dict[str, Any] | None,
        intent_id: str,
        recovered: bool = False,
    ) -> dict[str, Any]:
        """Record acceptance bookkeeping after every product write succeeded."""
        integration_id = str(run["integration_id"])
        # Mark artifacts accepted only after all product writes succeed.
        for item in run.get("applied_paths") or []:
            version_id = str(item["artifact_version_id"])
            state = self.store.read("artifact_states", version_id)
            if state.get("status") != "accepted":
                self.store.replace(
                    "artifact_states",
                    version_id,
                    {**state, "status": "accepted"},
                )

        post_tree = _product_revision(repo_root, excluded_root=self.store.root)
        applied_count = len(run.get("applied_paths") or [])
        if promotion is None:
            promotion_id = new_id("PROMO")
            promotion = {
                "schema_version": "cosmo.integration-promotion.v1",
                "promotion_id": promotion_id,
                "intent_id": intent_id,
                "integration_id": integration_id,
                "parent_node_id": run["parent_node_id"],
                "repo_root": str(repo_root),
                "baseline_tree_sha256": run.get("baseline_tree_sha256"),
                "post_tree_sha256": post_tree,
                "applied": [
                    {
                        "logical_path": item["logical_path"],
                        "artifact_version_id": item["artifact_version_id"],
                        "promoted_sha256": item["content_hash"],
                    }
                    for item in run.get("applied_paths") or []
                ],
                "created_at": utc_now(),
            }
            if recovered:
                promotion["recovered"] = True
            self.store.write("promotions", promotion_id, promotion)
        else:
            promotion_id = str(promotion["promotion_id"])

        updated = dict(run)
        updated["status"] = "promoted"
        updated["promotion_id"] = promotion_id
        updated["post_tree_sha256"] = post_tree
        updated["promoted_at"] = utc_now()
        self.store.replace("integration_runs", integration_id, updated)

        # Mark parent plan node approved.
        parent = self.store.read("plan_nodes", str(run["parent_node_id"]))
        parent["status"] = "approved"
        parent["integration_id"] = integration_id
        parent["promotion_id"] = promotion_id
        self.store.replace("plan_nodes", str(run["parent_node_id"]), parent)

        # Close the child lifecycle at the same acceptance boundary.  Without
        # this projection, reviewed/promoted leaves remain schedulable and can
        # be redispatched even though their bytes are already part of the
        # accepted parent integration.
        for selected in run.get("selected_children") or []:
            node_id = str(selected.get("node_id") or "")
            if not node_id or ":" in node_id:
                # Nested integration entries use a composed display id; the
                # nested parent run already owns their lifecycle.
                continue
            try:
                child_node = self.store.read("plan_nodes", node_id)
            except AtomicHarnessError:
                continue
            if child_node.get("status") not in {"canceled", "stale"}:
                child_node.update(
                    {
                        "status": "completed",
                        "integration_id": integration_id,
                        "accepted_artifact_version_id": selected.get(
                            "artifact_version_id"
                        ),
                        "completed_at": utc_now(),
                    }
                )
                self.store.replace("plan_nodes", node_id, child_node)
            task_id = child_node.get("task_id")
            if task_id:
                try:
                    child_task = self.store.read("tasks", str(task_id))
                    if child_task.get("status") not in {"canceled", "stale"}:
                        child_task["status"] = "completed"
                        child_task["updated_at"] = utc_now()
                        child_task["integration_id"] = integration_id
                        self.store.replace("tasks", str(task_id), child_task)
                except AtomicHarnessError:
                    pass

        # Complete parent task if present.
        parent_task_id = parent.get("task_id")
        if parent_task_id:
            try:
                task = self.store.read("tasks", str(parent_task_id))
                task["status"] = "completed"
                task["updated_at"] = utc_now()
                self.store.replace("tasks", str(parent_task_id), task)
            except AtomicHarnessError:
                pass

        intent = self.store.read("integration_promotion_intents", intent_id)
        self.store.replace(
            "integration_promotion_intents",
            intent_id,
            {**intent, "status": "committed", "promotion_id": promotion_id},
        )

        self.store.append_event(
            "integration_promotion_recovered" if recovered else "integration_promoted",
            {
                "integration_id": integration_id,
                "promotion_id": promotion_id,
                "parent_node_id": run["parent_node_id"],
                "applied_count": applied_count,
            },
        )
        return promotion

    def _promote_locked(
        self, run: Mapping[str, Any], *, repo_root: Path
    ) -> dict[str, Any]:
        integration_id = str(run["integration_id"])
        current_tree = _product_revision(repo_root, excluded_root=self.store.root)
        if current_tree != run.get("baseline_tree_sha256"):
            # Mark stale; do not mutate product.
            stale = dict(run)
            stale["status"] = "stale"
            stale["stale_reason"] = {
                "expected_tree": run.get("baseline_tree_sha256"),
                "observed_tree": current_tree,
            }
            self.store.replace("integration_runs", integration_id, stale)
            self.store.append_event(
                "integration_promotion_rejected_stale",
                {
                    "integration_id": integration_id,
                    "expected_tree": run.get("baseline_tree_sha256"),
                    "observed_tree": current_tree,
                },
            )
            raise AtomicStaleError(
                "accepted baseline changed before integration promotion"
            )

        # Stage and validate all bytes first so a later hash failure cannot
        # leave a partially-applied product tree. Rollback uses captured
        # previous bytes under the existing promotion lock.
        staged: list[dict[str, Any]] = []
        for item in run.get("applied_paths") or []:
            logical = _safe_relative_path(str(item["logical_path"]))
            artifact = self.store.read(
                "artifacts", str(item["artifact_version_id"])
            )
            content = Path(str(artifact["content_path"])).read_bytes()
            content_hash = _sha256_bytes(content)
            if content_hash != artifact.get("content_hash"):
                raise IntegrationError(
                    f"artifact content hash mismatch: {artifact['version_id']}"
                )
            if artifact.get("content_hash") != item.get("content_hash"):
                raise IntegrationError(
                    "integration applied_paths hash diverges from artifact store"
                )
            if content_hash != item.get("content_hash"):
                raise IntegrationError(
                    f"staged content hash mismatch for {logical}"
                )
            target = _artifact_target(repo_root, logical)
            previous: bytes | None = None
            previous_existed = target.is_file() and not target.is_symlink()
            if previous_existed:
                previous = target.read_bytes()
            staged.append(
                {
                    "logical_path": logical,
                    "artifact_version_id": artifact["version_id"],
                    "content_hash": content_hash,
                    "content": content,
                    "target": target,
                    "previous": previous,
                    "previous_existed": previous_existed,
                    "artifact": artifact,
                }
            )

        # Compute the exact post-write tree the promotion must produce, then
        # persist it as a durable prepared intent before touching product.
        baseline_snapshot = _product_snapshot(
            repo_root, excluded_root=self.store.root
        )
        after_tree = dict(baseline_snapshot)
        for entry in staged:
            after_tree[entry["logical_path"]] = entry["content_hash"]
        expected_post_tree = canonical_hash(after_tree)
        intent_id = self._prepare_promotion_intent(
            run,
            staged_entries=[
                {
                    "logical_path": entry["logical_path"],
                    "artifact_version_id": entry["artifact_version_id"],
                    "content_hash": entry["content_hash"],
                }
                for entry in staged
            ],
            repo_root=repo_root,
            post_tree_sha256=expected_post_tree,
        )

        committed: list[dict[str, Any]] = []
        try:
            for entry in staged:
                target: Path = entry["target"]
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".cosmo.tmp",
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(entry["content"])
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Re-check staged bytes before replace.
                    if _sha256_bytes(temporary.read_bytes()) != entry["content_hash"]:
                        raise IntegrationError(
                            f"staged temporary hash mismatch for "
                            f"{entry['logical_path']}"
                        )
                    os.replace(temporary, target)
                    directory_descriptor = os.open(
                        target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                    )
                    try:
                        os.fsync(directory_descriptor)
                    finally:
                        os.close(directory_descriptor)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                # Track product mutation immediately so rollback covers a failed
                # post-write hash check on this path.
                committed.append(entry)
                # Verify installed bytes before continuing.
                installed = target.read_bytes()
                if _sha256_bytes(installed) != entry["content_hash"]:
                    raise IntegrationError(
                        f"post-write hash check failed for {entry['logical_path']}"
                    )
            actual_post_tree = _product_revision(
                repo_root, excluded_root=self.store.root
            )
            if actual_post_tree != expected_post_tree:
                raise IntegrationError(
                    "product tree diverged during integration apply; "
                    "rolling back to the accepted baseline"
                )
        except Exception:
            # Roll back any committed product writes before re-raising.
            for entry in reversed(committed):
                target = entry["target"]
                try:
                    if entry["previous_existed"] and entry["previous"] is not None:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        descriptor, temporary_name = tempfile.mkstemp(
                            dir=target.parent,
                            prefix=f".{target.name}.",
                            suffix=".cosmo.rollback",
                        )
                        temporary = Path(temporary_name)
                        try:
                            with os.fdopen(descriptor, "wb") as stream:
                                stream.write(entry["previous"])
                                stream.flush()
                                os.fsync(stream.fileno())
                            os.replace(temporary, target)
                        finally:
                            if temporary.exists():
                                temporary.unlink()
                    elif target.exists():
                        target.unlink()
                except OSError:
                    # Best-effort rollback; still fail closed on the original error.
                    pass
            raise

        return self._complete_promotion(
            run,
            repo_root=repo_root,
            promotion=None,
            intent_id=intent_id,
            recovered=False,
        )
