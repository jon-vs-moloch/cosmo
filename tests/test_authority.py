"""Adversarial tests for the model/operator and repository-root boundaries."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cosmo.atomic import AtomicStore
from cosmo.authority import (
    AuthorityError,
    approve_repo_root,
    authenticate_model,
    authenticate_operator,
)
from cosmo.chat import ChatSession
from cosmo.control import ControlPlane
from cosmo.models import Generation
from cosmo.tools import (
    MODEL_REPO_ROOT_TOOLS,
    ToolDispatcher,
    tool_definitions_json,
)
from cosmo.triage import TriageService


class _ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def chat(self, **kwargs):
        return Generation(
            role=str(kwargs.get("role") or "chat"),
            content=self.responses.pop(0),
            reasoning_hash=None,
            stats={},
            latency_seconds=0.0,
            request_hash="authority-test",
            model_instance_id=kwargs.get("model"),
        )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    approved = tmp_path / "approved"
    outside = tmp_path / "outside"
    approved.mkdir()
    outside.mkdir()
    (approved / "inside.txt").write_text("inside\n", encoding="utf-8")
    (outside / "secret.txt").write_text("outside\n", encoding="utf-8")
    return approved, outside


def _model_dispatcher(
    tmp_path: Path,
) -> tuple[ToolDispatcher, ControlPlane, Path, Path]:
    approved, outside = _roots(tmp_path)
    plane = ControlPlane(AtomicStore(tmp_path / "state"))
    operator = authenticate_operator("test-operator")
    model = authenticate_model("test-model")
    capability = approve_repo_root(
        operator,
        repo_root=approved,
        grantee=model,
        allowed_tools=MODEL_REPO_ROOT_TOOLS,
    )
    return (
        ToolDispatcher(plane, caller=model, repo_capability=capability),
        plane,
        approved,
        outside,
    )


def test_model_catalog_has_no_operator_attestation_or_repo_root_arguments():
    catalog = {item["name"]: item for item in tool_definitions_json(catalog="model")}
    assert "create_task" not in catalog
    assert "recover_promotion" not in catalog
    for definition in catalog.values():
        assert "repo_root" not in definition["parameters"]["properties"]
    assert "proposal" not in catalog["propose_triage"]["parameters"]["properties"]
    assert "review" not in catalog["review_triage"]["parameters"]["properties"]
    assert "review" not in catalog["review_plan"]["parameters"]["properties"]
    integration = catalog["review_integration"]["parameters"]["properties"]
    assert {"verdict", "requirement_results", "notes"}.isdisjoint(integration)


def test_model_cannot_self_attest_operator_pass_or_use_operator_shortcut(tmp_path):
    dispatcher, plane, _approved, _outside = _model_dispatcher(tmp_path)
    request = plane.ingest_request(request_text="Do a bounded change.")

    proposal = dispatcher.execute(
        "propose_triage",
        {
            "request_id": request["request_id"],
            "proposal": {
                "relationship": {"type": "NEW"},
                "proposed_task_change": {"objective": "invented"},
                "priority_components": {},
                "confidence": 1.0,
                "open_questions": [],
            },
        },
    )
    assert proposal.ok is False
    assert "unknown tool arguments: proposal" in (proposal.error or "")

    pass_attempts = (
        (
            "review_triage",
            {
                "proposal_id": "TRIAGE-attacker",
                "review": {"verdict": "PASS", "evidence": ["self"]},
            },
            "review",
        ),
        (
            "review_plan",
            {
                "plan_version_id": "PLANVER-attacker",
                "review": {"verdict": "PASS", "evidence": ["self"]},
            },
            "review",
        ),
        (
            "review_integration",
            {"integration_id": "INTEG-attacker", "verdict": "PASS"},
            "verdict",
        ),
    )
    for tool_name, arguments, hidden_field in pass_attempts:
        attempted_pass = dispatcher.execute(tool_name, arguments)
        assert attempted_pass.ok is False
        assert f"unknown tool arguments: {hidden_field}" in (
            attempted_pass.error or ""
        )

    submit_shortcut = dispatcher.execute(
        "submit_request",
        {
            "request_text": "manufacture operator defaults",
            "objective": "forbidden shortcut",
            "artifact_path": "inside.txt",
        },
    )
    assert submit_shortcut.ok is False
    assert "unknown tool arguments" in (submit_shortcut.error or "")

    shortcut = dispatcher.execute(
        "create_task",
        {
            "request_text": "manufacture operator triage",
            "objective": "forbidden",
            "artifact_path": "inside.txt",
        },
    )
    assert shortcut.ok is False
    assert "unavailable" in (shortcut.error or "")
    assert plane.list_tasks() == []


def test_repo_capability_injects_one_root_and_rejects_override_or_escape(tmp_path):
    dispatcher, _plane, approved, outside = _model_dispatcher(tmp_path)

    inside = dispatcher.execute("read_artifact", {"path": "inside.txt"})
    assert inside.ok is True
    assert inside.result["excerpt"] == "inside\n"

    override = dispatcher.execute(
        "read_artifact",
        {"repo_root": str(outside), "path": "secret.txt"},
    )
    assert override.ok is False
    assert "unknown tool arguments: repo_root" in (override.error or "")

    traversal = dispatcher.execute(
        "read_artifact", {"path": "../outside/secret.txt"}
    )
    assert traversal.ok is False
    assert "normalized relative path" in (traversal.error or "")

    (approved / "secret-link").symlink_to(outside / "secret.txt")
    symlink_escape = dispatcher.execute(
        "read_artifact", {"path": "secret-link"}
    )
    assert symlink_escape.ok is False
    assert "symlink" in (symlink_escape.error or "")
    assert approved.resolve() != outside.resolve()


def test_repo_capability_is_immutable_bound_to_grantee_and_required(tmp_path):
    dispatcher, plane, approved, _outside = _model_dispatcher(tmp_path)
    capability = dispatcher.repo_capability
    assert capability is not None

    with pytest.raises(FrozenInstanceError):
        capability.repo_root = tmp_path  # type: ignore[misc]
    with pytest.raises(AttributeError):
        capability.allowed_tools.add("arbitrary_tool")  # type: ignore[attr-defined]

    thief = authenticate_model("different-model")
    stolen = ToolDispatcher(plane, caller=thief, repo_capability=capability)
    denied = stolen.execute("read_artifact", {"path": "inside.txt"})
    assert denied.ok is False
    assert "belongs to another caller" in (denied.error or "")

    same_label_new_session = ToolDispatcher(
        plane,
        caller=authenticate_model("test-model"),
        repo_capability=capability,
    )
    replay = same_label_new_session.execute(
        "read_artifact", {"path": "inside.txt"}
    )
    assert replay.ok is False
    assert "belongs to another caller" in (replay.error or "")

    ungranted = ToolDispatcher(plane, caller=authenticate_model("no-root"))
    missing = ungranted.execute("read_artifact", {"path": "inside.txt"})
    assert missing.ok is False
    assert "operator-approved repo-root capability" in (missing.error or "")

    # Optional legacy repo_root arguments must not fall back to a stored root.
    optional_root = ungranted.execute(
        "promote_integration", {"integration_id": "INTEG-attacker-selected"}
    )
    assert optional_root.ok is False
    assert "operator-approved repo-root capability" in (
        optional_root.error or ""
    )
    assert approved.is_dir()


def test_repo_capability_rejects_root_rebinding_after_grant(tmp_path):
    dispatcher, _plane, approved, outside = _model_dispatcher(tmp_path)
    moved = tmp_path / "approved-moved"
    approved.rename(moved)
    approved.symlink_to(outside, target_is_directory=True)

    rebound = dispatcher.execute("read_artifact", {"path": "secret.txt"})
    assert rebound.ok is False
    assert "root identity changed" in (rebound.error or "")
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "outside\n"


def test_operator_payload_requires_authenticated_operator_and_records_provenance(
    tmp_path,
):
    approved, _outside = _roots(tmp_path)
    plane = ControlPlane(AtomicStore(tmp_path / "state"))
    service = TriageService(plane)
    request = plane.ingest_request(request_text="Change inside.txt safely.")
    payload = {
        "relationship": {
            "type": "NEW",
            "target_task_id": None,
            "rationale": "new bounded task",
        },
        "proposed_task_change": {
            "title": "Bounded change",
            "summary": "Change the approved file",
            "added_requirements": ["Change inside.txt safely."],
            "modified_requirements": [],
            "inferred_requirements": [],
            "constraints": [],
            "plan_kind": "leaf",
            "artifact_path": "inside.txt",
            "objective": "Change inside.txt safely.",
        },
        "priority_components": {},
        "confidence": "high",
        "open_questions": [],
    }

    with pytest.raises(AuthorityError):
        service.propose_triage(
            request["request_id"],
            proposal=payload,
            operator_caller=authenticate_model("self-attesting-model"),
        )

    operator = authenticate_operator("reviewing-operator")
    record = service.propose_triage(
        request["request_id"], proposal=payload, operator_caller=operator
    )
    assert record["operator_supplied"] is True
    assert record["caller_provenance"] == operator.to_record()
    assert record["caller_provenance"]["principal_type"] == "operator"
    assert approved.is_dir()


def test_chat_keeps_explicit_capability_and_journals_authenticated_use(tmp_path):
    dispatcher, plane, _approved, _outside = _model_dispatcher(tmp_path)
    client = _ScriptedClient(
        [
            json.dumps(
                {
                    "tool_calls": [
                        {"name": "read_artifact", "arguments": {"path": "inside.txt"}}
                    ]
                }
            ),
            "Read the approved artifact.",
        ]
    )
    dispatcher.client = client
    dispatcher.model = "test-model"
    capability = dispatcher.repo_capability
    assert capability is not None

    session = ChatSession(
        client=client,
        dispatcher=dispatcher,
        model="test-model",
    )
    result = session.handle_user_message("Read the approved artifact.")
    assert result.stop_reason == "final"
    assert session.dispatcher.repo_capability is capability

    events = plane.store.list_records("chat_events")
    assert len(events) == 1
    event = events[0]
    assert event["caller_provenance"]["principal_type"] == "model"
    assert event["repo_capability_id"] == capability.capability_id
