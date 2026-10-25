"""Artifact-centric execution spine for Cosmo Slice 1.

The older ``run``/``evolve`` commands remain useful reasoning experiments, but
they are not the product architecture described by the Cosmo specification.
This module provides the smaller, stricter primitive that later triage,
planning, scheduling, and parent integration can build on:

    immutable request -> versioned task spec -> one-artifact attempt
    -> deterministic validation -> independent review -> explicit promotion

Product files are never touched by a Worker. Candidates and all evidence live
under the configured state root until a separately invoked promotion succeeds.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

from .inference import InferenceClient
from .models import Generation
from .reporting import canonical_hash, write_json


class AtomicHarnessError(RuntimeError):
    """A durable atomic-harness invariant or protocol failed."""


class AtomicStaleError(AtomicHarnessError):
    """The accepted baseline changed before a candidate was promoted."""


REVIEW_VERDICTS = {
    "PASS",
    "REVISE",
    "REPLAN",
    "BLOCKED",
    "STALE",
    "ESCALATE",
}

# Leaf validators are symbolic, dependency-free checks.  They are deliberately
# not shell commands: a plan may name these primitives, while broader test
# commands belong to the explicit post-integration validation step.
SUPPORTED_DETERMINISTIC_VALIDATORS = frozenset(
    {"non_empty", "utf8", "json", "python_syntax", "python"}
)

WORKER_SYSTEM = """You are a bounded Cosmo Worker.
You may propose exactly one artifact: the declared writable path. Read the task
contract and current artifact, then produce the complete replacement content.
Do not modify neighboring files, invent a multi-file patch, add Markdown
fences, or include a report. Return only the candidate artifact content."""


def _normalize_worker_artifact(content: str, artifact_path: str) -> tuple[str, dict[str, Any] | None]:
    """Remove one outer markdown fence a local coding model may add.

    The worker contract remains one complete artifact: patch encodings are
    rejected before persistence, while arbitrary prose, multiple fenced blocks,
    and status reports remain subject to the declared validators. This narrow
    transport normalization makes the protocol tolerant of the
    common `` ```python ... ``` `` presentation while recording that it
    happened in the attempt evidence.
    """
    text = content or ""
    match = re.fullmatch(
        r"\s*```(?P<language>[A-Za-z0-9_+.-]*)\s*\n?(?P<body>.*?)\s*```\s*",
        text,
        re.DOTALL,
    )
    if not match:
        body = text
        normalization = None
    else:
        body = match.group("body")
        normalization = {
            "kind": "outer_markdown_fence",
            "language": match.group("language") or None,
            "artifact_path": artifact_path,
        }
    if _looks_like_patch(body):
        raise AtomicHarnessError(
            "worker artifact must be complete replacement content, not a diff or patch"
        )
    if body and not body.endswith("\n"):
        body += "\n"
    return body, normalization


def _looks_like_patch(content: str) -> bool:
    """Reject patch encodings before candidate bytes enter the workspace.

    A Worker is allowed to return a complete replacement artifact only.  This
    catches the common unified-diff and apply-patch envelopes regardless of
    whether a Markdown fence was stripped first.  It intentionally treats a
    textual patch as a protocol error even when a weak leaf validator (such as
    ``non_empty``) would otherwise accept its bytes.
    """
    lines = [line.lstrip() for line in content.splitlines()]
    if any(
        line.startswith(("diff --git ", "*** Begin Patch", "*** Update File:"))
        for line in lines
    ):
        return True
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_hunk = any(line.startswith("@@ ") for line in lines)
    return has_old and has_new and has_hunk


REVIEW_SYSTEM = """You are an independent Cosmo Reviewer.
Evaluate the candidate against the supplied requirements and deterministic
validation evidence. Do not continue the Worker trajectory and do not propose
code edits yourself. Return exactly one JSON object matching the review schema.
Observations are evidence-backed facts. Hypotheses are explanations and must
remain explicitly typed as hypotheses. Every requirement_id must have a result
of PASS, FAIL, or BLOCKED and at least one concrete evidence string. A PASS
verdict requires PASS for every requirement. Keep the response compact: use
one short evidence sentence per requirement, empty arrays for optional
observation categories when there is nothing material to add, and no artifact
metadata or prose outside the JSON object."""

REVIEW_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cosmo_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": sorted(REVIEW_VERDICTS),
                },
                "requirement_results": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "requirement_id": {"type": "string"},
                            "result": {
                                "type": "string",
                                "enum": ["PASS", "FAIL", "BLOCKED"],
                            },
                            "evidence": {
                                "type": "array",
                                "minItems": 1,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["requirement_id", "result", "evidence"],
                        "additionalProperties": False,
                    },
                },
                "observations": {"type": "array"},
                "retained_successes": {"type": "array"},
                "hypotheses": {"type": "array"},
                "invalidated_assumptions": {"type": "array"},
                "recommended_state_changes": {"type": "array"},
                "recommended_next_action": {"type": "string"},
            },
            "required": [
                "verdict",
                "requirement_results",
                "observations",
                "retained_successes",
                "hypotheses",
                "invalidated_assumptions",
                "recommended_state_changes",
                "recommended_next_action",
            ],
            "additionalProperties": False,
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Allocate a time-ordered, collision-resistant durable record identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"


# Backwards-compatible private alias used by existing helpers in this module.
_new_id = new_id


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise AtomicHarnessError(
            f"writable artifact must be a normalized relative path: {value!r}"
        )
    return path.as_posix()


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise AtomicHarnessError(f"artifact path is not a file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AtomicHarnessError(f"artifact is not valid UTF-8: {path}") from exc


def _artifact_target(repo_root: Path, artifact_path: str) -> Path:
    """Resolve a declared path while refusing symlink/path escape."""
    root = repo_root.resolve()
    target = (root / artifact_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AtomicHarnessError(
            f"declared artifact escapes repo_root: {artifact_path}"
        ) from exc
    return target


def _assert_state_root_disjoint(repo_root: Path, state_root: Path) -> None:
    """Refuse a state root that would hide the approved product tree.

    Cosmo deliberately supports a state directory inside a repository (for
    example ``.cosmo``), but the inverse relationship is unsafe: if the state
    root is the repository itself or an ancestor of it, product snapshots and
    context inspection would exclude the whole product tree.  Keep this guard
    at the shared boundary so every product-facing entry point gets the same
    fail-closed behavior.
    """
    root = Path(repo_root).resolve()
    state = Path(state_root).resolve()
    try:
        root.relative_to(state)
    except ValueError:
        return
    raise AtomicHarnessError(
        "state_root must not be the repository root or an ancestor of it; "
        f"repo_root={root} state_root={state}"
    )


def _snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha256_bytes(
                path.read_bytes()
            )
    return result


def _product_snapshot(root: Path, *, excluded_root: Path) -> dict[str, str]:
    """Hash the product working tree without hashing Cosmo's own state.

    A state root is commonly placed at ``.cosmo`` inside a disposable fixture.
    It must not make the accepted product revision change merely because the
    harness appended an event.  Git's administrative directory is likewise
    excluded: accepted product state is the working tree, not repository
    internals.
    """
    root = root.resolve()
    excluded_root = excluded_root.resolve()
    _assert_state_root_disjoint(root, excluded_root)
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        # A symlink could point outside the approved product root.  Product
        # snapshots are a trust boundary, so refuse it instead of hashing
        # external bytes or making stale checks depend on outside state.
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        try:
            resolved.relative_to(excluded_root)
        except ValueError:
            pass
        else:
            continue
        result[relative.as_posix()] = _sha256_bytes(path.read_bytes())
    return result


def _product_revision(root: Path, *, excluded_root: Path) -> str:
    return canonical_hash(_product_snapshot(root, excluded_root=excluded_root))


def _safe_record_identifier(identifier: str) -> str:
    """Constrain a durable-record key to one filename component."""
    if (
        not identifier
        or identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
        or Path(identifier).name != identifier
    ):
        raise AtomicHarnessError(
            f"invalid atomic record identifier: {identifier!r}"
        )
    return identifier


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold a filesystem lock for a state-changing promotion operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parse_review(
    content: str, *, required_requirement_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    """Parse and validate the independent, evidence-backed review envelope."""
    text = (content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AtomicHarnessError(f"review output is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AtomicHarnessError("review output must be one JSON object")

    # LM Studio's native endpoint does not support response_format, so local
    # models occasionally emit a compact, semantically equivalent wrapper even
    # when the prompt asks for the canonical envelope. Normalize that one
    # observed shape before applying the strict schema/evidence checks below.
    # This is deliberately narrow: unknown wrappers still fail closed.
    if set(value) == {"review"} and isinstance(value.get("review"), dict):
        compact = value["review"]
        status = compact.get("status") or compact.get("verdict")
        findings = compact.get("findings") or compact.get("requirements")
        if isinstance(status, str) and isinstance(findings, list):
            normalized_findings: list[dict[str, Any]] = []
            for finding in findings:
                if not isinstance(finding, dict):
                    raise AtomicHarnessError(
                        "compact review findings must be objects"
                    )
                evidence = finding.get("evidence")
                if isinstance(evidence, str):
                    finding = {**finding, "evidence": [evidence]}
                normalized_findings.append(
                    {
                        "requirement_id": finding.get("requirement_id"),
                        "result": finding.get("result"),
                        "evidence": finding.get("evidence"),
                    }
                )
            value = {
                "verdict": status,
                "requirement_results": normalized_findings,
                "observations": compact.get("observations") or [],
                "retained_successes": compact.get("retained_successes") or [],
                "hypotheses": compact.get("hypotheses") or [],
                "invalidated_assumptions": compact.get("invalidated_assumptions") or [],
                "recommended_state_changes": compact.get("recommended_state_changes") or [],
                "recommended_next_action": (
                    compact.get("recommended_next_action")
                    or ("promote" if status == "PASS" else "retry")
                ),
            }
    elif isinstance(value.get("requirements"), (list, dict)) and isinstance(
        value.get("verdict") or value.get("result") or value.get("status"), str
    ):
        # Coding-oriented local models often use a descriptive envelope with
        # artifact hashes plus ``requirements`` instead of Cosmo's compact
        # requirement_results field.  Preserve only the semantically relevant
        # fields and normalize evidence strings; unknown metadata is ignored.
        normalized_findings = []
        raw_requirements = value["requirements"]
        findings_iter = (
            raw_requirements.items()
            if isinstance(raw_requirements, dict)
            else ((None, finding) for finding in raw_requirements)
        )
        for requirement_key, finding in findings_iter:
            if not isinstance(finding, dict):
                raise AtomicHarnessError(
                    "review requirements entries must be objects"
                )
            evidence = finding.get("evidence")
            if isinstance(evidence, str):
                evidence = [evidence]
            normalized_findings.append(
                {
                    "requirement_id": finding.get("requirement_id")
                    or requirement_key,
                    "result": finding.get("result"),
                    "evidence": evidence,
                }
            )
        status = value.get("verdict") or value.get("result") or value.get("status")
        value = {
            "verdict": status,
            "requirement_results": normalized_findings,
            "observations": value.get("observations") or [],
            "retained_successes": value.get("retained_successes") or [],
            "hypotheses": value.get("hypotheses") or [],
            "invalidated_assumptions": value.get("invalidated_assumptions") or [],
            "recommended_state_changes": value.get("recommended_state_changes") or [],
            "recommended_next_action": value.get(
                "recommended_next_action"
            ) or ("promote" if str(status).upper() == "PASS" else "retry"),
        }
    # Normalize short scalar observations/categories from local models.  The
    # content is still evidence-only; arbitrary objects remain invalid below.
    for list_key in (
        "observations",
        "retained_successes",
        "hypotheses",
        "invalidated_assumptions",
        "recommended_state_changes",
    ):
        if list_key in value and value[list_key] is None:
            value[list_key] = []
        elif isinstance(value.get(list_key), str):
            value[list_key] = [value[list_key]]
        elif isinstance(value.get(list_key), dict):
            value[list_key] = [
                json.dumps(value[list_key], sort_keys=True)
            ]
    required = {
        "verdict",
        "requirement_results",
        "observations",
        "retained_successes",
        "hypotheses",
        "invalidated_assumptions",
        "recommended_state_changes",
        "recommended_next_action",
    }
    if not required.issubset(value):
        raise AtomicHarnessError(
            "review output keys differ from the required review schema"
        )
    # Preserve the strict required fields while tolerating harmless artifact
    # metadata (review_id, candidate hashes, artifact_path) from coding models.
    if set(value) != required:
        value = {key: value[key] for key in required}
    verdict = value["verdict"]
    if verdict not in REVIEW_VERDICTS:
        raise AtomicHarnessError(f"unsupported review verdict: {verdict!r}")
    for key in required - {"verdict", "recommended_next_action"}:
        if not isinstance(value[key], list):
            raise AtomicHarnessError(f"review field must be an array: {key}")
    if not isinstance(value["recommended_next_action"], str):
        raise AtomicHarnessError("review next action must be a string")
    requirement_results = value["requirement_results"]
    if not requirement_results:
        raise AtomicHarnessError("review must assess at least one requirement")
    observed_ids: set[str] = set()
    for item in requirement_results:
        if not isinstance(item, dict) or set(item) != {
            "requirement_id",
            "result",
            "evidence",
        }:
            raise AtomicHarnessError(
                "each requirement result must contain only requirement_id, "
                "result, and evidence"
            )
        requirement_id = item["requirement_id"]
        result = item["result"]
        evidence = item["evidence"]
        if not isinstance(requirement_id, str) or not requirement_id.strip():
            raise AtomicHarnessError("requirement_id must be a non-empty string")
        if requirement_id in observed_ids:
            raise AtomicHarnessError(
                f"review repeats requirement_id: {requirement_id}"
            )
        observed_ids.add(requirement_id)
        if result not in {"PASS", "FAIL", "BLOCKED"}:
            raise AtomicHarnessError(
                f"unsupported requirement result: {result!r}"
            )
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
        ):
            raise AtomicHarnessError(
                f"requirement {requirement_id} needs non-empty evidence"
            )
    required_ids = {
        str(item) for item in (required_requirement_ids or ()) if str(item).strip()
    }
    if required_ids and observed_ids != required_ids:
        raise AtomicHarnessError(
            "review requirement coverage differs from the TaskSpec: "
            f"expected {sorted(required_ids)}, observed {sorted(observed_ids)}"
        )
    if value["verdict"] == "PASS" and any(
        item["result"] != "PASS" for item in requirement_results
    ):
        raise AtomicHarnessError(
            "PASS review cannot contain a failed or blocked requirement"
        )
    return value


def deterministic_validate(
    *,
    content: str,
    artifact_path: str,
    validators: Iterable[str],
) -> dict[str, Any]:
    """Run the dependency-free validators declared by a leaf plan."""
    names = list(dict.fromkeys(validators)) or ["non_empty", "utf8"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "name": name,
                "result": "pass" if passed else "fail",
                "evidence": evidence,
            }
        )

    for name in names:
        if name == "non_empty":
            check(name, bool(content), "candidate contains at least one character")
        elif name == "utf8":
            try:
                content.encode("utf-8")
            except UnicodeEncodeError as exc:
                check(name, False, str(exc))
            else:
                check(name, True, "candidate encoded as UTF-8")
        elif name == "json":
            try:
                json.loads(content)
            except (json.JSONDecodeError, ValueError) as exc:
                check(name, False, str(exc))
            else:
                check(name, True, "candidate parses as JSON")
        elif name in {"python_syntax", "python"}:
            try:
                ast.parse(content, filename=artifact_path)
            except SyntaxError as exc:
                check(name, False, str(exc))
            else:
                check(name, True, "candidate parses as Python syntax")
        else:
            check(name, False, f"unknown deterministic validator: {name}")
    return {
        "passed": all(item["result"] == "pass" for item in checks),
        "checks": checks,
        "validator_names": names,
        "candidate_sha256": _sha256_text(content),
    }


class AtomicStore:
    """Filesystem-backed immutable records and append-only event history."""

    KINDS = (
        "requests",
        # Request records are immutable; lifecycle/status fields live in this
        # separate derived projection and never rewrite the request body.
        "request_states",
        "tasks",
        "task_specs",
        "plan_nodes",
        "triage_proposals",
        "triage_reviews",
        "plan_reviews",
        "plan_versions",
        "relationships",
        "priority_assessments",
        "leases",
        "context_manifests",
        "attempts",
        "attempt_states",
        "artifacts",
        "artifact_states",
        "validations",
        "reviews",
        "promotion_intents",
        "promotions",
        "convergence_assessments",
        "ledger_projections",
        "chat_events",
        "ledger",
        "workspaces",
        "inference_journal",
        "worker_tool_events",
        "integration_runs",
        "integration_reviews",
        "integration_promotion_intents",
    )

    def __init__(self, root: Path):
        self.root = root.resolve()
        for kind in self.KINDS:
            (self.root / kind).mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, identifier: str) -> Path:
        if kind not in self.KINDS:
            raise AtomicHarnessError(f"unknown atomic record kind: {kind}")
        safe_identifier = _safe_record_identifier(identifier)
        return self.root / kind / f"{safe_identifier}.json"

    def write(self, kind: str, identifier: str, value: dict[str, Any]) -> None:
        path = self.path(kind, identifier)
        if path.exists():
            raise AtomicHarnessError(f"immutable record already exists: {path}")
        write_json(path, value)

    def replace(self, kind: str, identifier: str, value: dict[str, Any]) -> None:
        write_json(self.path(kind, identifier), value)

    def read(self, kind: str, identifier: str) -> dict[str, Any]:
        path = self.path(kind, identifier)
        if not path.is_file():
            raise AtomicHarnessError(f"record not found: {kind}/{identifier}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AtomicHarnessError(f"invalid record: {path}") from exc
        if not isinstance(value, dict):
            raise AtomicHarnessError(f"record is not an object: {path}")
        return value

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        path = self.root / "events.jsonl"
        record = {"event_id": _new_id("EVT"), "type": event_type, "at": utc_now()}
        record.update(payload)
        write_json(self.root / "ledger" / f"{record['event_id']}.json", record)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def list_records(self, kind: str) -> list[dict[str, Any]]:
        if kind not in self.KINDS:
            raise AtomicHarnessError(f"unknown atomic record kind: {kind}")
        directory = self.root / kind
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]

    def promotion_lock(self, repo_root: Path) -> Path:
        digest = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()
        return self.root / "locks" / f"{digest}.lock"


class AtomicHarness:
    """Request-to-promotion lifecycle for one atomic artifact leaf."""

    def __init__(self, store: AtomicStore):
        self.store = store

    def submit(
        self,
        *,
        request_text: str,
        objective: str,
        artifact_path: str,
        repo_root: Path,
        validators: Iterable[str],
        desired_outcome: str | None = None,
        explicit_constraints: Iterable[str] = (),
    ) -> dict[str, Any]:
        if not request_text.strip():
            raise AtomicHarnessError("request_text must not be empty")
        if not objective.strip():
            raise AtomicHarnessError("objective must not be empty")
        repo_root = repo_root.resolve()
        if not repo_root.is_dir():
            raise AtomicHarnessError(f"repo_root is not a directory: {repo_root}")
        safe_path = _safe_relative_path(artifact_path)
        constraints = list(explicit_constraints)
        baseline_content = _read_text(_artifact_target(repo_root, safe_path))
        baseline_sha256 = (
            _sha256_text(baseline_content) if baseline_content is not None else None
        )
        baseline_tree_sha256 = _product_revision(
            repo_root, excluded_root=self.store.root
        )
        request_id = _new_id("REQ")
        task_id = _new_id("TASK")
        spec_id = _new_id("SPEC")
        node_id = _new_id("NODE")
        request = {
            "schema_version": "cosmo.request.v1",
            "request_id": request_id,
            "original_text": request_text,
            "desired_outcome": desired_outcome,
            "explicit_constraints": constraints,
            "created_at": utc_now(),
            "immutable_sha256": canonical_hash(
                {
                    "original_text": request_text,
                    "desired_outcome": desired_outcome,
                    "explicit_constraints": constraints,
                }
            ),
        }
        spec = {
            "schema_version": "cosmo.task-spec.v1",
            "spec_id": spec_id,
            "task_id": task_id,
            "version": 1,
            "objective": objective,
            "requirements": [
                {"requirement_id": "objective", "text": objective}
            ],
            "acceptance_criteria": [
                f"the declared artifact exists at {safe_path}",
                "all declared deterministic validators pass",
                "an independent review returns PASS",
            ],
            "constraints": constraints,
            "artifact_path": safe_path,
            "artifact_type": "source_file" if safe_path.endswith(".py") else "artifact",
            "validators": list(dict.fromkeys(validators)) or ["non_empty", "utf8"],
            "repo_root": str(repo_root.resolve()),
            "baseline_artifact_sha256": baseline_sha256,
            "baseline_tree_sha256": baseline_tree_sha256,
            "created_at": utc_now(),
        }
        task = {
            "schema_version": "cosmo.task.v1",
            "task_id": task_id,
            "status": "ready",
            "linked_request_ids": [request_id],
            "current_spec_id": spec_id,
            "current_spec_version": 1,
            "plan_node_id": node_id,
            "created_at": utc_now(),
        }
        plan = {
            "schema_version": "cosmo.plan-node.v1",
            "node_id": node_id,
            "task_id": task_id,
            "kind": "leaf",
            "status": "approved",
            "objective": objective,
            "writable_artifact": safe_path,
            "validators": spec["validators"],
            "dependencies": [],
            "parent_node_id": None,
            "created_at": utc_now(),
        }
        triage_proposal = {
            "schema_version": "cosmo.triage-proposal.v1",
            "proposal_id": _new_id("TRIAGE"),
            "request_id": request_id,
            "relationship": {"type": "NEW", "target_task_id": None},
            "proposed_task_id": task_id,
            "mode": "operator_supplied_leaf_contract",
            "created_at": utc_now(),
        }
        triage_review = {
            "schema_version": "cosmo.triage-review.v1",
            "review_id": _new_id("TRIAGE-REVIEW"),
            "proposal_id": triage_proposal["proposal_id"],
            "verdict": "PASS",
            "independent_reviewer": "operator_leaf_contract",
            "created_at": utc_now(),
        }
        plan_review = {
            "schema_version": "cosmo.plan-review.v1",
            "review_id": _new_id("PLAN-REVIEW"),
            "node_id": node_id,
            "verdict": "PASS",
            "checks": [
                "exactly one writable artifact",
                "local deterministic validators declared",
            ],
            "created_at": utc_now(),
        }
        self.store.write("requests", request_id, request)
        self.store.write("tasks", task_id, task)
        self.store.write("task_specs", spec_id, spec)
        self.store.write("plan_nodes", node_id, plan)
        self.store.write(
            "triage_proposals", triage_proposal["proposal_id"], triage_proposal
        )
        self.store.write("triage_reviews", triage_review["review_id"], triage_review)
        self.store.write("plan_reviews", plan_review["review_id"], plan_review)
        self.store.append_event(
            "task_submitted",
            {"request_id": request_id, "task_id": task_id, "spec_id": spec_id},
        )
        return {
            "request_id": request_id,
            "task_id": task_id,
            "spec_id": spec_id,
            "node_id": node_id,
            "artifact_path": safe_path,
            "state_root": str(self.store.root),
        }

    def _task_context(self, task_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        task = self.store.read("tasks", task_id)
        spec = self.store.read("task_specs", str(task["current_spec_id"]))
        return task, spec

    def _request_context(self, task: dict[str, Any]) -> dict[str, Any]:
        request_ids = task.get("linked_request_ids") or []
        if not request_ids:
            raise AtomicHarnessError("Task has no linked immutable Request")
        return self.store.read("requests", str(request_ids[0]))

    def _read_attempt(self, attempt_id: str) -> dict[str, Any]:
        base = self.store.read("attempts", attempt_id)
        state_path = self.store.path("attempt_states", attempt_id)
        if not state_path.is_file():
            return base
        state = self.store.read("attempt_states", attempt_id)
        return {**base, **state}

    def _update_attempt(self, attempt: dict[str, Any]) -> None:
        """Update a derived projection without mutating the immutable Attempt."""
        self.store.replace("attempt_states", str(attempt["attempt_id"]), attempt)

    def _prior_attempts(self, task_id: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self.store.list_records("attempts")
            if item.get("task_id") == task_id
        ]

    def _dependency_excerpts(
        self,
        *,
        repo_root: Path,
        read_only_paths: Iterable[str] | None,
    ) -> list[dict[str, Any]]:
        """Load small, bounded read-only dependency excerpts for worker context.

        Paths must stay under the approved product root and outside the
        AtomicStore state root. Whole-repo dumps are never accepted here.
        """
        if not read_only_paths:
            return []
        # Late import avoids circular import with context -> atomic.
        from .context import RepositoryInspector

        inspector = RepositoryInspector(repo_root, state_root=self.store.root)
        dependencies: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in read_only_paths:
            path = _safe_relative_path(str(raw))
            if path in seen:
                continue
            seen.add(path)
            try:
                payload = inspector.read_artifact(path)
            except Exception as exc:  # ContextError or path issues
                raise AtomicHarnessError(
                    f"read_only_paths entry is not readable under repo_root: "
                    f"{path}: {exc}"
                ) from exc
            dependencies.append(
                {
                    "kind": "dependency",
                    "path": payload["path"],
                    "source_path": payload["path"],
                    "content_hash": payload.get("content_sha256")
                    or payload.get("sha256"),
                    "sha256": payload.get("sha256"),
                    "selection_method": "dependency",
                    "excerpt": payload.get("excerpt"),
                    "included_range_or_excerpt": {
                        "range": payload.get("range"),
                        "excerpt": payload.get("excerpt"),
                        "truncated": payload.get("truncated"),
                    },
                    "metadata": {
                        "size_bytes": payload.get("size_bytes"),
                        "language": payload.get("language"),
                    },
                }
            )
        return dependencies

    def _create_attempt(
        self,
        *,
        task_id: str,
        repo_root: Path,
        strategy_summary: str,
        retry_contract: dict[str, Any] | None,
        read_only_paths: Iterable[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None, Path, dict[str, Any]]:
        task, spec = self._task_context(task_id)
        request = self._request_context(task)
        if task.get("status") not in {"ready", "queued", "running", "executing"}:
            raise AtomicHarnessError(
                "cannot start an attempt for task in terminal or stale status: "
                f"{task.get('status')!r}"
            )
        if not strategy_summary.strip():
            raise AtomicHarnessError("strategy_summary must not be empty")
        if repo_root.resolve() != Path(str(spec["repo_root"])).resolve():
            raise AtomicHarnessError(
                "repo_root differs from the task specification; create a new "
                "TaskSpec rather than silently changing the baseline"
            )
        artifact_path = _safe_relative_path(str(spec["artifact_path"]))
        product_path = _artifact_target(repo_root, artifact_path)
        baseline = _read_text(product_path)
        baseline_sha = _sha256_text(baseline) if baseline is not None else None
        baseline_tree_sha = _product_revision(
            repo_root, excluded_root=self.store.root
        )
        if (
            baseline_sha != spec.get("baseline_artifact_sha256")
            or baseline_tree_sha != spec.get("baseline_tree_sha256")
        ):
            raise AtomicHarnessError(
                "product baseline differs from the TaskSpec; create a new "
                "TaskSpec rather than starting a stale attempt"
            )
        prior = self._prior_attempts(task_id)
        if prior:
            required = {
                "addresses",
                "preserves",
                "differs_from",
                "expected_improvement",
            }
            if not isinstance(retry_contract, dict) or any(
                not retry_contract.get(key) for key in required
            ):
                raise AtomicHarnessError(
                    "retries require addresses, preserves, differs_from, and "
                    "expected_improvement metadata"
                )
            strategy_key = strategy_summary.strip()
            if not strategy_key:
                raise AtomicHarnessError("strategy_summary must not be empty")
            if any(
                strategy_key == str(item.get("strategy_summary", "")).strip()
                for item in prior
            ):
                raise AtomicHarnessError(
                    "retry strategy is identical to a prior attempt; provide a "
                    "materially different strategy"
                )
            prior_ids = {str(item["attempt_id"]) for item in prior}
            differences = [str(item) for item in retry_contract["differs_from"]]
            if not any(
                any(prior_id in difference for prior_id in prior_ids)
                for difference in differences
            ):
                raise AtomicHarnessError(
                    "differs_from must reference at least one prior attempt id"
                )

        # Rebuild a compact, durable convergence projection before assembling
        # the next Worker context.  This keeps rejected strategies and review
        # observations visible to retries without replaying prior transcripts.
        # The import is intentionally local: ControlPlane is layered on top of
        # AtomicHarness and importing it at module load would create a cycle.
        from .control import ControlPlane

        ledger_projection = ControlPlane(self.store).project_ledger(task_id)
        attempt_id = _new_id("ATTEMPT")
        workspace = self.store.root / "workspaces" / attempt_id
        writable_path = workspace / artifact_path
        writable_path.parent.mkdir(parents=True, exist_ok=True)
        if baseline is not None:
            writable_path.write_text(baseline, encoding="utf-8")

        # Canonical worker context via ContextCompiler (immutable worker schema).
        from .context import ContextCompiler

        dependencies = self._dependency_excerpts(
            repo_root=repo_root, read_only_paths=read_only_paths
        )
        writable_item: dict[str, Any] = {
            "path": artifact_path,
            "source_path": artifact_path,
            "content_hash": baseline_sha,
            "sha256": baseline_sha,
            "excerpt": baseline,
            "included_range_or_excerpt": baseline,
            "metadata": {"role": "writable", "baseline_tree_sha256": baseline_tree_sha},
        }
        context = ContextCompiler(self.store).compile_for_worker(
            task_spec=spec,
            writable_artifact=writable_item,
            read_only_dependencies=dependencies,
            strategy_summary=strategy_summary,
            ledger_projection=ledger_projection,
            request=request,
        )
        # Preserve legacy aliases used by existing event/callers.
        context_id = str(context["context_manifest_id"])
        context.setdefault("context_id", context_id)
        context.setdefault("task_id", task_id)
        context.setdefault("spec_id", spec["spec_id"])
        # Worker instruction hash is part of the attempt contract, not the
        # durable context items body (avoids competing context shapes).
        attempt = {
            "schema_version": "cosmo.attempt.v1",
            "attempt_id": attempt_id,
            "task_id": task_id,
            "node_id": task["plan_node_id"],
            "spec_id": spec["spec_id"],
            "task_spec_sha256": canonical_hash(spec),
            "request_id": request["request_id"],
            "baseline_revision": baseline_sha,
            "baseline_artifact_sha256": baseline_sha,
            "baseline_tree_sha256": baseline_tree_sha,
            "context_manifest_id": context_id,
            "context_manifest_sha256": context.get("manifest_sha256"),
            "ledger_projection_id": ledger_projection.get("projection_id"),
            "ledger_projection_sha256": canonical_hash(ledger_projection),
            "writable_artifact": artifact_path,
            "repo_root": str(repo_root.resolve()),
            "workspace": str(workspace),
            "strategy_summary": strategy_summary,
            "strategy_sha256": _sha256_text(strategy_summary),
            "retry_contract": retry_contract,
            "read_only_paths": [
                str(item["path"]) for item in dependencies
            ],
            "worker_protocol": {
                "output_mode": "complete_artifact_content",
                "declared_artifact": artifact_path,
                "metadata_recorded_by_harness": True,
            },
            "status": "created",
            "created_at": utc_now(),
        }
        self.store.write("attempts", attempt_id, attempt)
        self.store.write("attempt_states", attempt_id, attempt)
        self.store.append_event(
            "attempt_created",
            {
                "attempt_id": attempt_id,
                "task_id": task_id,
                "context_id": context_id,
                "context_manifest_id": context_id,
                "manifest_sha256": context.get("manifest_sha256"),
            },
        )
        return attempt, spec, baseline, writable_path, context

    def run_attempt(
        self,
        *,
        task_id: str,
        repo_root: Path,
        client: InferenceClient,
        model: str,
        reasoning: str,
        temperature: float,
        worker_max_output_tokens: int,
        review_max_output_tokens: int,
        strategy_summary: str,
        retry_contract: dict[str, Any] | None = None,
        read_only_paths: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        attempt, spec, baseline, writable_path, context = self._create_attempt(
            task_id=task_id,
            repo_root=repo_root,
            strategy_summary=strategy_summary,
            retry_contract=retry_contract,
            read_only_paths=read_only_paths,
        )
        attempt_id = str(attempt["attempt_id"])
        artifact_path = str(spec["artifact_path"])
        task, _ = self._task_context(task_id)
        request = self._request_context(task)
        current = baseline if baseline is not None else "(artifact does not yet exist)"
        dependency_block = ""
        if attempt.get("read_only_paths"):
            dep_items = [
                item
                for item in (context.get("items") or [])
                if item.get("kind") == "dependency"
                or item.get("selection_method") == "dependency"
            ]
            dependency_block = f"""
READ-ONLY DEPENDENCY EXCERPTS
{json.dumps(dep_items, indent=2, sort_keys=True)}
"""
        worker_prompt = f"""TASK OBJECTIVE
{spec['objective']}

ORIGINAL REQUEST
{request['original_text']}

DESIRED OUTCOME
{request.get('desired_outcome') or '(not separately specified)'}

TASK SPECIFICATION
spec_id: {spec['spec_id']}
version: {spec['version']}

REQUIREMENTS
{json.dumps(spec['requirements'], indent=2)}

EXPLICIT CONSTRAINTS
{json.dumps(spec.get('constraints') or [], indent=2)}

ACCEPTANCE CRITERIA
{json.dumps(spec['acceptance_criteria'], indent=2)}

WRITABLE ARTIFACT
{artifact_path}

CURRENT ARTIFACT
{current}

DETERMINISTIC VALIDATORS
{json.dumps(spec['validators'])}

STRATEGY NOTE
{strategy_summary}

CONTEXT MANIFEST
context_manifest_id: {context.get('context_manifest_id')}
role: {context.get('role')}
manifest_sha256: {context.get('manifest_sha256')}
items: {json.dumps(context.get('items') or [], indent=2, sort_keys=True)}
excluded: {json.dumps(context.get('excluded') or [], indent=2)}
{dependency_block}
OUTPUT PROTOCOL
Return only the complete replacement content for {artifact_path}; do not return
Markdown fences, a patch, a second file, a status report, or review JSON. The
harness records Worker status and generation metadata separately from the one
artifact content."""
        try:
            generation = client.chat(
                model=model,
                role="atomic_worker",
                system_prompt=WORKER_SYSTEM,
                prompt=worker_prompt,
                reasoning=reasoning,
                temperature=temperature,
                max_output_tokens=worker_max_output_tokens,
                operation_id=f"atomic/{attempt_id}/worker",
            )
        except Exception as exc:
            attempt["status"] = "worker_failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            self._update_attempt(attempt)
            self.store.append_event(
                "worker_failed", {"attempt_id": attempt_id, "error": str(exc)}
            )
            raise

        workspace_root = Path(str(attempt["workspace"]))
        before = _snapshot(workspace_root)
        candidate, normalization = _normalize_worker_artifact(
            generation.content, artifact_path
        )
        if normalization is not None:
            attempt["worker_protocol_normalization"] = normalization
        try:
            # The canonical executor must enforce the same bounded output and
            # write policy that WorkerToolset advertises, before any bytes are
            # persisted.  The late policy attachment in WorkerExecutor is
            # evidence-only; this is the actual safety gate.
            from .sandbox import build_worker_policy

            policy = build_worker_policy(
                repo_root=repo_root,
                state_root=self.store.root,
                attempt_workspace=workspace_root,
                writable_artifact=artifact_path,
            )
            policy.assert_output_size(candidate)
            policy.assert_write_size(candidate)
            writable_path.write_text(candidate, encoding="utf-8")
            candidate = _read_text(writable_path)
        except (OSError, UnicodeError, AtomicHarnessError) as exc:
            attempt["status"] = "worker_protocol_failed"
            attempt["error"] = f"candidate artifact could not be written: {exc}"
            self._update_attempt(attempt)
            self.store.append_event(
                "worker_protocol_failed",
                {"attempt_id": attempt_id, "error": str(exc)},
            )
            raise AtomicHarnessError(str(exc)) from exc
        if candidate is None:
            raise AtomicHarnessError("candidate artifact disappeared from workspace")
        after = _snapshot(workspace_root)
        changed = sorted(
            path
            for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        )
        if not changed:
            attempt["status"] = "no_op"
            attempt["write_scope"] = {
                "declared": artifact_path,
                "changed_paths": [],
                "passed": False,
            }
            self._update_attempt(attempt)
            self.store.append_event(
                "attempt_rejected_no_op", {"attempt_id": attempt_id}
            )
            raise AtomicHarnessError(
                "attempt produced no artifact change; an explicit blocked/no-op "
                "outcome is required instead"
            )
        if changed != [artifact_path]:
            attempt["status"] = "invalid_write_scope"
            attempt["write_scope"] = {
                "declared": artifact_path,
                "changed_paths": changed,
                "passed": False,
            }
            self._update_attempt(attempt)
            self.store.append_event(
                "attempt_rejected_write_scope",
                {"attempt_id": attempt_id, "changed_paths": changed},
            )
            raise AtomicHarnessError(
                f"attempt changed undeclared paths: {', '.join(changed)}"
            )

        version_id = _new_id("ARTV")
        content_hash = _sha256_text(candidate)
        content_path = self.store.root / "artifacts" / f"{content_hash}.content"
        if not content_path.exists():
            _write_text_atomic(content_path, candidate)
        artifact = {
            "schema_version": "cosmo.artifact-version.v1",
            "artifact_id": f"ART-{content_hash[:16]}",
            "logical_path": artifact_path,
            "version_id": version_id,
            "content_hash": content_hash,
            "content_path": str(content_path),
            "created_by_attempt": attempt_id,
            "baseline_revision": attempt["baseline_revision"],
            "task_spec_version": spec["spec_id"],
            "status": "candidate",
            "created_at": utc_now(),
        }
        self.store.write("artifacts", version_id, artifact)
        self.store.write("artifact_states", version_id, artifact)
        validation_id = _new_id("VAL")
        validation = deterministic_validate(
            content=candidate,
            artifact_path=artifact_path,
            validators=spec["validators"],
        )
        validation.update(
            {
                "schema_version": "cosmo.validation.v1",
                "validation_id": validation_id,
                "attempt_id": attempt_id,
                "artifact_version_id": version_id,
                "baseline_revision": attempt["baseline_revision"],
                "validator_version": "cosmo-slice1-v1",
                "environment_hash": canonical_hash(
                    {"python": os.sys.version, "validators": spec["validators"]}
                ),
                "created_at": utc_now(),
            }
        )
        self.store.write("validations", validation_id, validation)
        self.store.append_event(
            "validation_completed",
            {
                "attempt_id": attempt_id,
                "validation_id": validation_id,
                "passed": validation["passed"],
            },
        )

        review_prompt = f"""ORIGINAL REQUEST
{request['original_text']}

DESIRED OUTCOME
{request.get('desired_outcome') or '(not separately specified)'}

TASK SPECIFICATION
{json.dumps(spec, indent=2, sort_keys=True)}

CONTEXT MANIFEST
{json.dumps(self.store.read('context_manifests', str(attempt['context_manifest_id'])), indent=2, sort_keys=True)}

WORKER REPORT
{json.dumps({'strategy_summary': strategy_summary, 'generation_stats': generation.stats}, indent=2, sort_keys=True)}

CANDIDATE ARTIFACT ({artifact_path})
{candidate}

DETERMINISTIC VALIDATION
{json.dumps(validation, indent=2, sort_keys=True)}

Review the candidate against every requirement_id. Each requirement result must
contain concrete evidence references (for example validation check names or
candidate hashes). Return the required JSON object and nothing else."""
        try:
            review_generation = client.chat(
                model=model,
                role="atomic_reviewer",
                system_prompt=REVIEW_SYSTEM,
                prompt=review_prompt,
                reasoning=reasoning,
                temperature=0,
                max_output_tokens=review_max_output_tokens,
                response_format=REVIEW_RESPONSE_FORMAT,
                operation_id=f"atomic/{attempt_id}/review",
            )
            review = parse_review(
                review_generation.content,
                required_requirement_ids=[
                    str(item["requirement_id"])
                    for item in spec["requirements"]
                ],
            )
        except Exception as exc:
            attempt.update(
                {
                    "status": "review_failed",
                    "artifact_version_id": version_id,
                    "validation_id": validation_id,
                    "worker_generation": asdict(generation),
                    "review_generation": (
                        asdict(review_generation)
                        if "review_generation" in locals()
                        else None
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._update_attempt(attempt)
            self.store.append_event(
                "review_failed", {"attempt_id": attempt_id, "error": str(exc)}
            )
            raise
        review_id = _new_id("REVIEW")
        review_record = {
            "schema_version": "cosmo.review.v1",
            "review_id": review_id,
            "attempt_id": attempt_id,
            "artifact_version_id": version_id,
            "validation_id": validation_id,
            "reviewer_role": "atomic_reviewer",
            "generation": asdict(review_generation),
            **review,
            "created_at": utc_now(),
        }
        self.store.write("reviews", review_id, review_record)
        attempt.update(
            {
                "status": "reviewed",
                "artifact_version_id": version_id,
                "validation_id": validation_id,
                "review_id": review_id,
                "write_scope": {
                    "declared": artifact_path,
                    "changed_paths": changed,
                    "passed": not changed or changed == [artifact_path],
                },
                "worker_generation": asdict(generation),
                "review_generation": asdict(review_generation),
                "finished_at": utc_now(),
            }
        )
        self._update_attempt(attempt)
        if validation.get("passed") and review.get("verdict") == "PASS":
            # Node approval is the durable handoff from leaf review to parent
            # integration.  It is distinct from product promotion and prevents
            # the scheduler from redispatching already accepted bytes.
            try:
                node = self.store.read("plan_nodes", str(task["plan_node_id"]))
                if node.get("kind") == "leaf" and node.get("status") not in {
                    "stale",
                    "canceled",
                }:
                    node.update(
                        {
                            "status": "node_approved",
                            "accept_node_approved": True,
                            "approved_attempt_id": attempt_id,
                            "approved_artifact_version_id": version_id,
                            "approved_validation_id": validation_id,
                            "approved_review_id": review_id,
                            "node_approved_at": utc_now(),
                        }
                    )
                    self.store.replace("plan_nodes", str(node["node_id"]), node)
            except AtomicHarnessError:
                # Missing plan projections are already a control-plane error;
                # preserve the reviewed attempt evidence for legacy callers.
                pass
        self.store.append_event(
            "review_completed",
            {
                "attempt_id": attempt_id,
                "review_id": review_id,
                "verdict": review["verdict"],
                "validation_passed": validation["passed"],
            },
        )
        return {
            "attempt_id": attempt_id,
            "artifact_version_id": version_id,
            "validation_id": validation_id,
            "review_id": review_id,
            "validation_passed": validation["passed"],
            "review_verdict": review["verdict"],
            "promotable": bool(validation["passed"] and review["verdict"] == "PASS"),
            "state_root": str(self.store.root),
        }

    def promote(self, *, attempt_id: str, repo_root: Path) -> dict[str, Any]:
        """Promote under a per-repository lock and revalidate current state."""
        with _exclusive_lock(self.store.promotion_lock(repo_root)):
            recovered = self._recover_prepared_intent(
                attempt_id=attempt_id, repo_root=repo_root
            )
            if recovered is not None:
                return recovered
            return self._promote_locked(attempt_id=attempt_id, repo_root=repo_root)

    def _recover_prepared_intent(
        self, *, attempt_id: str, repo_root: Path
    ) -> dict[str, Any] | None:
        """Finalize a promotion whose product replace completed before a crash."""
        intents = [
            item
            for item in self.store.list_records("promotion_intents")
            if item.get("attempt_id") == attempt_id
            and item.get("status") == "prepared"
        ]
        if not intents:
            return None
        intent = intents[-1]
        if Path(str(intent.get("repo_root"))).resolve() != repo_root.resolve():
            raise AtomicHarnessError(
                "promotion repo_root differs from the prepared intent"
            )
        target = _artifact_target(repo_root, str(intent["logical_path"]))
        current = _read_text(target)
        current_hash = _sha256_text(current) if current is not None else None
        current_tree = _product_revision(repo_root, excluded_root=self.store.root)
        if (
            current_hash != intent.get("promoted_sha256")
            or current_tree != intent.get("post_tree_sha256")
        ):
            return None
        attempt = self._read_attempt(attempt_id)
        if attempt.get("status") != "reviewed":
            return None
        artifact = self.store.read("artifacts", str(intent["artifact_version_id"]))
        existing = [
            item
            for item in self.store.list_records("promotions")
            if item.get("attempt_id") == attempt_id
        ]
        if existing:
            promotion = existing[-1]
            self.store.replace(
                "artifact_states",
                str(artifact["version_id"]),
                {
                    **self.store.read(
                        "artifact_states", str(artifact["version_id"])
                    ),
                    "status": "accepted",
                },
            )
            attempt["status"] = "promoted"
            attempt["promotion_id"] = promotion["promotion_id"]
            self._update_attempt(attempt)
            task, _spec = self._task_context(str(attempt["task_id"]))
            task["status"] = "completed"
            self.store.replace("tasks", str(task["task_id"]), task)
            return promotion
        promotion_id = _new_id("PROMO")
        promotion = {
            "schema_version": "cosmo.promotion.v1",
            "promotion_id": promotion_id,
            "intent_id": intent["intent_id"],
            "attempt_id": attempt_id,
            "artifact_version_id": artifact["version_id"],
            "logical_path": artifact["logical_path"],
            "baseline_sha256": attempt.get("baseline_artifact_sha256"),
            "baseline_tree_sha256": attempt.get("baseline_tree_sha256"),
            "post_tree_sha256": intent["post_tree_sha256"],
            "promoted_sha256": artifact["content_hash"],
            "repo_root": str(repo_root.resolve()),
            "recovered": True,
            "created_at": utc_now(),
        }
        self.store.write("promotions", promotion_id, promotion)
        self.store.replace(
            "artifact_states",
            str(artifact["version_id"]),
            {**self.store.read("artifact_states", str(artifact["version_id"])), "status": "accepted"},
        )
        attempt["status"] = "promoted"
        attempt["promotion_id"] = promotion_id
        self._update_attempt(attempt)
        task, _spec = self._task_context(str(attempt["task_id"]))
        task["status"] = "completed"
        self.store.replace("tasks", str(task["task_id"]), task)
        self.store.append_event(
            "artifact_promotion_recovered",
            {
                "promotion_id": promotion_id,
                "intent_id": intent["intent_id"],
                "attempt_id": attempt_id,
            },
        )
        return promotion

    def _promote_locked(self, *, attempt_id: str, repo_root: Path) -> dict[str, Any]:
        attempt = self._read_attempt(attempt_id)
        if attempt.get("status") != "reviewed":
            raise AtomicHarnessError("only a reviewed attempt can be promoted")
        validation = self.store.read("validations", str(attempt["validation_id"]))
        review = self.store.read("reviews", str(attempt["review_id"]))
        if not validation.get("passed") or review.get("verdict") != "PASS":
            raise AtomicHarnessError(
                "promotion requires deterministic validation PASS and review PASS"
            )
        if repo_root.resolve() != Path(str(attempt["repo_root"])).resolve():
            raise AtomicHarnessError("promotion repo_root differs from attempt baseline")
        task, spec = self._task_context(str(attempt["task_id"]))
        if (
            task.get("current_spec_id") != attempt.get("spec_id")
            or canonical_hash(spec) != attempt.get("task_spec_sha256")
        ):
            attempt["status"] = "stale"
            attempt["stale_reason"] = "TaskSpec changed after attempt"
            self._update_attempt(attempt)
            self.store.append_event(
                "promotion_rejected_stale",
                {"attempt_id": attempt_id, "reason": "task_spec_changed"},
            )
            raise AtomicStaleError("TaskSpec changed before promotion")
        target = _artifact_target(repo_root, str(attempt["writable_artifact"]))
        current = _read_text(target)
        current_hash = _sha256_text(current) if current is not None else None
        current_tree_sha = _product_revision(
            repo_root, excluded_root=self.store.root
        )
        if (
            current_hash != attempt.get("baseline_artifact_sha256")
            or current_tree_sha != attempt.get("baseline_tree_sha256")
        ):
            attempt["status"] = "stale"
            attempt["stale_reason"] = {
                "expected_artifact": attempt.get("baseline_artifact_sha256"),
                "observed_artifact": current_hash,
                "expected_tree": attempt.get("baseline_tree_sha256"),
                "observed_tree": current_tree_sha,
            }
            self._update_attempt(attempt)
            self.store.append_event(
                "promotion_rejected_stale",
                {
                    "attempt_id": attempt_id,
                    "expected_artifact": attempt.get("baseline_artifact_sha256"),
                    "observed_artifact": current_hash,
                    "expected_tree": attempt.get("baseline_tree_sha256"),
                    "observed_tree": current_tree_sha,
                },
            )
            raise AtomicStaleError("accepted baseline changed before promotion")
        artifact = self.store.read("artifacts", str(attempt["artifact_version_id"]))
        content = Path(str(artifact["content_path"])).read_bytes()
        if _sha256_bytes(content) != artifact.get("content_hash"):
            raise AtomicHarnessError("candidate artifact content hash mismatch")
        if validation.get("candidate_sha256") != artifact.get("content_hash"):
            raise AtomicHarnessError(
                "validation does not cover the candidate artifact bytes"
            )
        before_tree = _product_snapshot(
            repo_root, excluded_root=self.store.root
        )
        after_tree = dict(before_tree)
        after_tree[str(artifact["logical_path"])] = artifact["content_hash"]
        prepared = [
            item
            for item in self.store.list_records("promotion_intents")
            if item.get("attempt_id") == attempt_id
            and item.get("status") == "prepared"
        ]
        if prepared:
            intent = prepared[-1]
            intent_id = str(intent["intent_id"])
            if (
                intent.get("artifact_version_id") != artifact["version_id"]
                or intent.get("promoted_sha256") != artifact["content_hash"]
            ):
                raise AtomicHarnessError(
                    "prepared promotion intent does not match the candidate"
                )
        else:
            intent_id = _new_id("PINTENT")
            intent = {
                "schema_version": "cosmo.promotion-intent.v1",
                "intent_id": intent_id,
                "attempt_id": attempt_id,
                "artifact_version_id": artifact["version_id"],
                "logical_path": artifact["logical_path"],
                "baseline_tree_sha256": attempt.get("baseline_tree_sha256"),
                "post_tree_sha256": canonical_hash(after_tree),
                "promoted_sha256": artifact["content_hash"],
                "repo_root": str(repo_root.resolve()),
                "status": "prepared",
                "created_at": utc_now(),
            }
            self.store.write("promotion_intents", intent_id, intent)
            self.store.append_event(
                "promotion_prepared",
                {"intent_id": intent_id, "attempt_id": attempt_id},
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".cosmo.tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
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
        installed = _read_text(target)
        if installed is None or _sha256_text(installed) != artifact.get("content_hash"):
            raise AtomicHarnessError(
                "post-write hash check failed for promoted artifact"
            )
        promotion_id = _new_id("PROMO")
        promotion = {
            "schema_version": "cosmo.promotion.v1",
            "promotion_id": promotion_id,
            "intent_id": intent_id,
            "attempt_id": attempt_id,
            "artifact_version_id": artifact["version_id"],
            "logical_path": artifact["logical_path"],
            "baseline_sha256": attempt.get("baseline_artifact_sha256"),
            "baseline_tree_sha256": attempt.get("baseline_tree_sha256"),
            "post_tree_sha256": intent["post_tree_sha256"],
            "promoted_sha256": artifact["content_hash"],
            "repo_root": str(repo_root.resolve()),
            "created_at": utc_now(),
        }
        self.store.write("promotions", promotion_id, promotion)
        artifact_state = {
            **self.store.read("artifact_states", str(artifact["version_id"])),
            "status": "accepted",
        }
        self.store.replace(
            "artifact_states", str(artifact["version_id"]), artifact_state
        )
        intent = self.store.read("promotion_intents", intent_id)
        self.store.replace(
            "promotion_intents",
            intent_id,
            {**intent, "status": "committed", "promotion_id": promotion_id},
        )
        attempt["status"] = "promoted"
        attempt["promotion_id"] = promotion_id
        self._update_attempt(attempt)
        task, _spec = self._task_context(str(attempt["task_id"]))
        task["status"] = "completed"
        self.store.replace("tasks", str(task["task_id"]), task)
        self.store.append_event(
            "artifact_promoted",
            {
                "promotion_id": promotion_id,
                "attempt_id": attempt_id,
                "artifact_version_id": artifact["version_id"],
            },
        )
        return promotion

    def show(self, identifier: str) -> dict[str, Any]:
        for kind in self.store.KINDS:
            path = self.store.path(kind, identifier)
            if path.is_file():
                record = self.store.read(kind, identifier)
                if kind == "attempts":
                    state_path = self.store.path("attempt_states", identifier)
                    if state_path.is_file():
                        record = {
                            **record,
                            **self.store.read("attempt_states", identifier),
                        }
                elif kind == "artifacts":
                    state_path = self.store.path("artifact_states", identifier)
                    if state_path.is_file():
                        record = {
                            **record,
                            **self.store.read("artifact_states", identifier),
                        }
                return {"kind": kind, "record": record}
        for kind in ("requests", "tasks", "task_specs", "attempts", "promotions"):
            records = self.store.list_records(kind)
            matches = [item for item in records if identifier in json.dumps(item)]
            if matches:
                return {"kind": kind, "records": matches}
        raise AtomicHarnessError(f"no atomic record matches: {identifier}")
