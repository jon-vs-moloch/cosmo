"""Focused tests for local-only PlanningService."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmo.atomic import AtomicStore
from cosmo.authority import authenticate_operator
from cosmo.control import ControlPlane
from cosmo.models import Generation
from cosmo.planning import (
    PlanningError,
    PlanningService,
    parse_plan_proposal,
    validate_plan_proposal,
)


_OPERATOR = authenticate_operator("pytest-operator")


def _proposal(
    *,
    leaf_a="src/a.py",
    leaf_b="src/b.py",
    bad_path=None,
):
    children = [
        {
            "temp_id": "c1",
            "objective": "Write module A",
            "kind": "leaf",
            "writable_artifact": bad_path or leaf_a,
            "depends_on": [],
            "validators": ["python_syntax"],
            "requirements": ["REQ-1"],
        },
        {
            "temp_id": "c2",
            "objective": "Write module B",
            "kind": "leaf",
            "writable_artifact": leaf_b,
            "depends_on": ["c1"],
            "validators": ["python_syntax"],
            "requirements": ["REQ-2"],
        },
    ]
    return {
        "objective": "Ship A and B",
        "requirements": ["REQ-1", "REQ-2"],
        "shared_contracts": [{"artifact": "docs/contract.md"}],
        "children": children,
        "parent_validators": ["compose_ok"],
        "risks": ["ordering"],
        "unresolved_questions": [],
    }


class FakePlannerClient:
    def __init__(self, plan=None, review=None, malformed=False):
        self.plan = plan if plan is not None else _proposal()
        self.review = review or {
            "verdict": "PASS",
            "evidence": ["one artifact per leaf", "deps ordered"],
            "notes": "ok",
        }
        self.malformed = malformed
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["role"] == "planner":
            content = "NOT JSON {" if self.malformed else json.dumps(self.plan)
        else:
            content = json.dumps(self.review)
        return Generation(
            role=kwargs["role"],
            content=content,
            reasoning_hash=None,
            stats={"input_tokens": 3, "total_output_tokens": 4, "reasoning_output_tokens": 0},
            latency_seconds=0.01,
            request_hash=f"req-{len(self.calls)}",
            model_instance_id=kwargs["model"],
        )


def _plane(tmp_path: Path) -> tuple[ControlPlane, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("A=1\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("B=1\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "contract.md").write_text("# contract\n", encoding="utf-8")
    state = tmp_path / "state"
    return ControlPlane(AtomicStore(state)), repo, state


def test_parse_plan_proposal_and_malformed():
    ok = parse_plan_proposal(json.dumps(_proposal()))
    assert ok["proposal_sha256"]
    assert ok["children"][0]["writable_artifact"] == "src/a.py"
    fenced = parse_plan_proposal("```json\n" + json.dumps(_proposal()) + "\n```")
    assert fenced["proposal_sha256"] == ok["proposal_sha256"]
    with pytest.raises(PlanningError):
        parse_plan_proposal("not-json")
    for malformed_wrapper in (
        "Here is the plan: " + json.dumps(_proposal()),
        json.dumps(_proposal()) + "\nThanks.",
        json.dumps(_proposal()) + "\n{}",
    ):
        with pytest.raises(PlanningError, match="one JSON object"):
            parse_plan_proposal(malformed_wrapper)
    with pytest.raises(PlanningError):
        validate_plan_proposal({"objective": "x"})
    with pytest.raises(PlanningError):
        validate_plan_proposal(
            {
                **_proposal(),
                "children": [
                    {
                        "temp_id": "c1",
                        "objective": "bad",
                        "kind": "leaf",
                        "writable_artifact": "../escape.py",
                        "depends_on": [],
                    }
                ],
            }
        )
    with pytest.raises(PlanningError, match="unsupported deterministic validators"):
        validate_plan_proposal(
            {
                **_proposal(),
                "children": [
                    {
                        "temp_id": "c1",
                        "objective": "bad validator",
                        "kind": "leaf",
                        "writable_artifact": "src/a.py",
                        "depends_on": [],
                        "validators": ["python -m pytest"],
                    }
                ],
            }
        )
    with pytest.raises(PlanningError):
        validate_plan_proposal(
            {
                **_proposal(),
                "children": [
                    {
                        "temp_id": "c1",
                        "objective": "a",
                        "kind": "leaf",
                        "writable_artifact": "src/a.py",
                        "depends_on": ["missing"],
                        "validators": ["python_syntax"],
                    }
                ],
            }
        )
    with pytest.raises(PlanningError):
        validate_plan_proposal(
            {
                **_proposal(leaf_a="src/same.py", leaf_b="src/same.py"),
            }
        )


def test_plan_review_context_caps_excerpts_and_retains_manifest_link(tmp_path):
    plane, _repo, state = _plane(tmp_path)
    store = AtomicStore(state)
    manifest_id = "CTX-projection-test"
    store.write(
        "context_manifests",
        manifest_id,
        {
            "context_manifest_id": manifest_id,
            "manifest_sha256": "manifest-hash",
            "role": "planner",
            "items": [
                {
                    "kind": "source",
                    "path": f"src/{index}.py",
                    "excerpt": "x" * 1000,
                }
                for index in range(7)
            ],
        },
    )
    projection = PlanningService(plane)._plan_review_context(
        {
            "context_manifest_id": manifest_id,
            "context_manifest_sha256": "manifest-hash",
            "proposal": _proposal(),
        }
    )
    excerpts = [item.get("excerpt", "") for item in projection["manifest"]["items"]]
    assert sum(map(len, excerpts)) <= 3500
    assert all(len(excerpt) <= 600 for excerpt in excerpts)
    assert projection["context_projection"]["complete_context_is_hash_linked_in_atomic_store"]
    assert projection["context_manifest_sha256"] == "manifest-hash"
    assert projection["context_projection"]["omitted_excerpt_items"] == 1


def test_planning_pack_and_propose_without_product_mutation(tmp_path):
    plane, repo, state = _plane(tmp_path)
    service = PlanningService(plane)
    before = (repo / "src" / "a.py").read_text(encoding="utf-8")
    client = FakePlannerClient()
    # Existing task optional.
    submitted = plane.submit_request(
        request_text="Ship modules",
        objective="Coordinate A/B",
        artifact_path="",
        repo_root=repo,
        plan_kind="parent",
    )
    version = service.propose_plan(
        repo_root=repo,
        client=client,
        model="local-planner",
        task_id=submitted["task_id"],
        selected_paths=["docs/contract.md"],
    )
    assert version["status"] == "proposed"
    assert version["approved"] is False
    assert version["applied"] is False
    assert version["proposal_sha256"]
    assert version["context_manifest_id"]
    assert version["generation"]["content_sha256"]
    assert "prompt" not in version["generation"]
    assert client.calls[0]["role"] == "planner"
    assert client.calls[0]["response_format"]["json_schema"]["name"] == (
        "cosmo_plan_proposal"
    )
    # Product tree untouched.
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == before
    # Malformed model output.
    bad = FakePlannerClient(malformed=True)
    with pytest.raises(PlanningError):
        service.propose_plan(
            repo_root=repo,
            client=bad,
            model="local-planner",
            task_id=submitted["task_id"],
            include_repository_map=False,
        )


def test_independent_review_and_approval_only_apply(tmp_path):
    plane, repo, state = _plane(tmp_path)
    service = PlanningService(plane)
    client = FakePlannerClient()
    version = service.propose_plan(
        repo_root=repo,
        client=client,
        model="local-planner",
        include_repository_map=True,
    )
    plan_version_id = version["plan_version_id"]

    # Unreviewed plan cannot be applied.
    with pytest.raises(PlanningError):
        service.apply_approved_plan(plan_version_id, repo_root=repo)

    # Rejected review still does not apply.
    reject = service.review_plan(
        plan_version_id,
        review={
            "verdict": "REJECT",
            "evidence": ["missing consumer leaf"],
            "notes": "incomplete",
        },
        operator_caller=_OPERATOR,
    )
    assert reject["verdict"] == "REJECT"
    with pytest.raises(PlanningError):
        service.apply_approved_plan(plan_version_id, repo_root=repo)

    # Fresh proposal + PASS review + apply.
    version2 = service.store_plan_proposal(
        proposal=_proposal(),
        pack={
            "task_id": None,
            "request_id": None,
            "spec_id": None,
            "context_manifest_id": version["context_manifest_id"],
            "context_manifest_sha256": version["context_manifest_sha256"],
            "pack_sha256": version["planning_pack_sha256"],
            "repository_map_sha256": version.get("repository_map_sha256"),
        },
        operator_supplied=True,
        operator_caller=_OPERATOR,
    )
    pass_review = service.review_plan(
        version2["plan_version_id"],
        review={
            "verdict": "PASS",
            "evidence": [
                "requirement coverage complete",
                "one writable artifact per leaf",
                "dependency order valid",
            ],
            "notes": "approved",
        },
        operator_caller=_OPERATOR,
    )
    assert pass_review["verdict"] == "PASS"
    applied = service.apply_approved_plan(
        version2["plan_version_id"], repo_root=repo
    )
    assert applied["parent_node_id"]
    assert len(applied["temp_id_map"]) == 2
    parent = plane.get_plan(applied["parent_node_id"])
    assert parent["kind"] == "parent"
    children = [
        plane.get_plan(nid) for nid in applied["temp_id_map"].values()
    ]
    leaves = [node for node in children if node["kind"] == "leaf"]
    assert len(leaves) == 2
    assert {node["writable_artifact"] for node in leaves} == {"src/a.py", "src/b.py"}
    root_request_id = plane.get_task(parent["task_id"])["linked_request_ids"][0]
    assert all(
        plane.get_task(node["task_id"])["linked_request_ids"] == [root_request_id]
        for node in leaves
    )
    assert plane.get_task_spec(plane.get_task(leaves[0]["task_id"])["current_spec_id"])[
        "requirements"
    ][0]["requirement_id"] in {"REQ-1", "REQ-2"}
    # Dependency preserved: c2 depends on c1's node id.
    c1 = applied["temp_id_map"]["c1"]
    c2 = applied["temp_id_map"]["c2"]
    assert c1 in plane.get_plan(c2)["dependencies"]
    # Parent lists children.
    for node_id in (c1, c2):
        assert node_id in plane.get_plan(applied["parent_node_id"])["child_node_ids"]
    # Already applied is idempotent-fail.
    with pytest.raises(PlanningError):
        service.apply_approved_plan(version2["plan_version_id"], repo_root=repo)
    # Product files still original (planning does not write product).
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == "A=1\n"


def test_model_backed_review_path(tmp_path):
    plane, repo, _state = _plane(tmp_path)
    service = PlanningService(plane)
    client = FakePlannerClient(
        review={
            "verdict": "REVISE",
            "evidence": ["need test leaf"],
            "notes": "add tests",
        }
    )
    version = service.propose_plan(
        repo_root=repo, client=client, model="local-planner"
    )
    review = service.review_plan(
        version["plan_version_id"],
        client=client,
        model="local-reviewer",
    )
    assert review["verdict"] == "REVISE"
    assert review["operator_supplied"] is False
    assert review["generation"]["content_sha256"]
    stored = service.get_plan_version(version["plan_version_id"])
    assert stored["approved"] is False


def test_parent_leaf_dependency_creation_from_nested_shape(tmp_path):
    plane, repo, _ = _plane(tmp_path)
    service = PlanningService(plane)
    nested = {
        "objective": "Nested plan",
        "requirements": ["R1"],
        "shared_contracts": [],
        "children": [
            {
                "temp_id": "leaf1",
                "objective": "Leaf one",
                "kind": "leaf",
                "writable_artifact": "src/a.py",
                "depends_on": [],
                "validators": ["python_syntax"],
                "requirements": ["R1"],
            },
            {
                "temp_id": "leaf2",
                "objective": "Leaf two",
                "kind": "leaf",
                "writable_artifact": "src/b.py",
                "depends_on": ["leaf1"],
                "validators": ["non_empty"],
            },
        ],
        "parent_validators": ["integration"],
        "risks": [],
        "unresolved_questions": ["none"],
    }
    # Build a real context via pack so hashes exist.
    built = service.build_planning_pack(repo_root=repo, include_repository_map=False)
    version = service.store_plan_proposal(
        proposal=nested,
        pack=built["pack"],
        operator_supplied=True,
        operator_caller=_OPERATOR,
    )
    service.review_plan(
        version["plan_version_id"],
        review={
            "verdict": "PASS",
            "evidence": ["ok"],
            "notes": "",
        },
        operator_caller=_OPERATOR,
    )
    result = service.apply_approved_plan(version["plan_version_id"], repo_root=repo)
    assert set(result["temp_id_map"]) == {"leaf1", "leaf2"}
    leaf2 = plane.get_plan(result["temp_id_map"]["leaf2"])
    assert result["temp_id_map"]["leaf1"] in leaf2["dependencies"]
    # Blocked until dependency completes.
    task2 = plane.get_task(leaf2["task_id"])
    assert task2["status"] == "blocked"


def test_nested_child_retains_dependency_on_uncle(tmp_path):
    """A nested leaf depending on a top-level sibling must keep the edge."""
    plane, repo, _ = _plane(tmp_path)
    (repo / "src" / "c.py").write_text("C=1\n", encoding="utf-8")
    service = PlanningService(plane)
    proposal = {
        "objective": "Nested with cross-level dep",
        "requirements": ["R1"],
        "shared_contracts": [],
        "children": [
            {
                "temp_id": "uncle",
                "objective": "Write A first",
                "kind": "leaf",
                "writable_artifact": "src/a.py",
                "depends_on": [],
                "validators": ["python_syntax"],
                "requirements": ["R1"],
            },
            {
                "temp_id": "mid",
                "objective": "Coordinate nested work",
                "kind": "parent",
                "depends_on": [],
                "validators": ["compose"],
                "children": [
                    {
                        "temp_id": "nested_leaf",
                        "objective": "Write B after uncle",
                        "kind": "leaf",
                        "writable_artifact": "src/b.py",
                        "depends_on": ["uncle"],
                        "validators": ["python_syntax"],
                    },
                    {
                        "temp_id": "nested_leaf2",
                        "objective": "Write C after nested B",
                        "kind": "leaf",
                        "writable_artifact": "src/c.py",
                        "depends_on": ["nested_leaf"],
                        "validators": ["python_syntax"],
                    },
                ],
            },
        ],
        "parent_validators": ["integration"],
        "risks": [],
        "unresolved_questions": [],
    }
    assert validate_plan_proposal(proposal)["proposal_sha256"]
    built = service.build_planning_pack(repo_root=repo, include_repository_map=False)
    version = service.store_plan_proposal(
        proposal=proposal,
        pack=built["pack"],
        operator_supplied=True,
        operator_caller=_OPERATOR,
    )
    service.review_plan(
        version["plan_version_id"],
        review={
            "verdict": "PASS",
            "evidence": ["nested deps valid", "one artifact per leaf"],
            "notes": "",
        },
        operator_caller=_OPERATOR,
    )
    result = service.apply_approved_plan(version["plan_version_id"], repo_root=repo)
    assert set(result["temp_id_map"]) == {"uncle", "mid", "nested_leaf", "nested_leaf2"}
    uncle_id = result["temp_id_map"]["uncle"]
    mid_id = result["temp_id_map"]["mid"]
    nested_id = result["temp_id_map"]["nested_leaf"]
    nested2_id = result["temp_id_map"]["nested_leaf2"]
    nested = plane.get_plan(nested_id)
    assert uncle_id in nested["dependencies"]
    assert nested["parent_node_id"] == mid_id
    nested2 = plane.get_plan(nested2_id)
    assert nested_id in nested2["dependencies"]
    assert nested2["parent_node_id"] == mid_id
    mid = plane.get_plan(mid_id)
    assert nested_id in mid["child_node_ids"]
    assert nested2_id in mid["child_node_ids"]
    # Product tree still untouched by planning apply.
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == "A=1\n"


def test_plan_cycle_and_duplicate_leaf_rejected():
    with pytest.raises(PlanningError, match="cycle"):
        validate_plan_proposal(
            {
                "objective": "cycle",
                "requirements": [],
                "children": [
                    {
                        "temp_id": "a",
                        "objective": "A",
                        "kind": "leaf",
                        "writable_artifact": "src/a.py",
                        "depends_on": ["b"],
                        "validators": ["non_empty"],
                    },
                    {
                        "temp_id": "b",
                        "objective": "B",
                        "kind": "leaf",
                        "writable_artifact": "src/b.py",
                        "depends_on": ["a"],
                        "validators": ["non_empty"],
                    },
                ],
                "parent_validators": [],
                "risks": [],
                "unresolved_questions": [],
            }
        )
