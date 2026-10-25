"""Small, model-readable Cosmo protocol guide.

The chat system prompt contains the tool catalog, but a bounded local model
also needs a cheap way to ask for the lifecycle and output contracts without
replaying a large transcript.  This module is the single source for the
model-visible ``harness_help`` tool and the human ``/help`` shortcut.
"""

from __future__ import annotations

import json
from typing import Any


HARNESS_HELP: dict[str, Any] = {
    "schema_version": "cosmo.harness-help.v1",
    "identity": "Cosmo is a local-only control-plane harness for atomic artifact work.",
    "lifecycle": [
        "broad outcome: ingest_request -> propose_triage -> review_triage -> apply_triage",
        "multi-artifact task: build_planning_pack -> propose_plan -> review_plan -> apply_approved_plan",
        "leaf execution: ready_queue/scheduler -> run_attempt -> validation + independent review",
        "parent completion: assemble_integration -> review_integration -> promote_integration",
    ],
    "plan_shape": {
        "leaf": "one writable artifact and one narrow responsibility",
        "parent": "coordination node with child leaves; no direct artifact write",
        "needs_planning": "triage says decomposition is required before execution",
        "important": "plan_kind is decomposition shape, not work type; implementation is a work type.",
    },
    "contracts": {
        "chat_turn": {
            "final": "plain user-facing text",
            "tool_call": {"tool_calls": [{"name": "<tool>", "arguments": {}}]},
        },
        "plan": "one JSON object with objective, requirements, children, parent_validators, risks, unresolved_questions",
        "model_json_transport": "exactly one JSON object, optionally wrapped by one complete ```json fence; no prose, prefix, suffix, or second object",
        "leaf": "exactly one writable_artifact; validators are symbolic: non_empty, utf8, json, python_syntax, python",
        "worker": "complete replacement content for the declared artifact only; one outer Markdown fence is normalized, but diffs/patches, reports, and second files are rejected",
        "review": "every requirement_id gets PASS, FAIL, or BLOCKED evidence; PASS requires all PASS",
    },
    "boundaries": [
        "models cannot promote product state or manufacture PASS evidence",
        "product files change only after explicit promotion",
        "repository tools stay under the operator-approved root capability",
        "workers can read bounded dependencies but write one declared artifact",
        "broader pytest/external acceptance runs are explicit post-integration validation, not leaf validator commands",
    ],
}


def harness_help_payload() -> dict[str, Any]:
    """Return a fresh JSON-safe help payload."""
    return json.loads(json.dumps(HARNESS_HELP, sort_keys=True))


def format_harness_help() -> str:
    """Format the compact guide for a human ``/help`` response."""
    return (
        "Cosmo help\n"
        "- Lifecycle: "
        + " -> ".join(
            item.split(": ", 1)[1] if ": " in item else item
            for item in HARNESS_HELP["lifecycle"]
        )
        + "\n"
        "- Plan shape: leaf=one artifact; parent=coordination; "
        "needs_planning=decompose first. 'implementation' is a work type, "
        "not a plan shape.\n"
        "- Worker: return one complete declared artifact. Reviews and promotion "
        "are separate gates.\n"
        "- Ask the model to call harness_help for the structured contracts."
    )
