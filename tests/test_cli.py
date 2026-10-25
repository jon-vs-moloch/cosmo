import argparse
import hashlib
import json
from types import SimpleNamespace

import pytest

from cosmo.cli import (
    _client,
    _reasoning_probe,
    compare,
    evolve,
    parser,
    resolve_chat_input_mode,
    run,
)
from cosmo.inference import InferenceError
from cosmo.models import Generation


def _manifest(run_id, arms):
    return {
        "schema_version": "cosmo.run.v3",
        "cosmo_version": "0.2.0",
        "run_id": run_id,
        "benchmark_sha256": "bench",
        "case_ids": ["case-1"],
        "model": "model",
        "model_instance_id": "instance",
        "model_record_sha256": "record",
        "model_artifact_sha256": "artifact",
        "chat_template_sha256": "template",
        "lmstudio": {"short_version": "1"},
        "source": {"commit": "abc", "dirty": False},
        "revision_protocol": "direct-adversarial-revision-v3",
        "prompt_hashes": {
            "draft": "d",
            "revision_system": "s",
            "revision_instruction": "r",
        },
        "evaluator_sha256": "evaluator",
        "capability_probe_validated": True,
        "base_url": "http://localhost",
        "max_cycles": 4,
        "temperature": 0,
        "max_output_tokens": 100,
        "revision_max_output_tokens": 100,
        "order_seed": 7,
        "store": False,
        "hidden_evaluation_is_terminal": True,
        "initial_draft_policy": "shared by case and reasoning",
        "arms": arms,
    }


def _write_run(root, run_id, arms, write_results=True):
    run_dir = root / run_id
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(_manifest(run_id, arms)), encoding="utf-8"
    )
    seed_by_reasoning = {}
    for arm in arms:
        reasoning = "on" if arm.startswith("thinking_") else "off"
        candidate = f"seed-{reasoning}"
        candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
        seed_by_reasoning[reasoning] = {
            "case_id": "case-1",
            "reasoning": reasoning,
            "candidate": candidate,
            "candidate_hash": candidate_hash,
            "generation": {"request_hash": f"request-{reasoning}"},
        }
    (run_dir / "seed_drafts.json").write_text(
        json.dumps(
            {
                "schema_version": "cosmo.seed-drafts.v1",
                "records": list(seed_by_reasoning.values()),
            }
        ),
        encoding="utf-8",
    )
    if write_results:
        for arm in arms:
            reasoning = "on" if arm.startswith("thinking_") else "off"
            seed = seed_by_reasoning[reasoning]
            (run_dir / "results" / f"case-1--{arm}.json").write_text(
                json.dumps(
                    {
                        "case_id": "case-1",
                        "family": "test",
                        "arm_id": arm,
                        "reasoning": reasoning,
                        "recursive": False,
                        "stop_reason": "one_shot",
                        "error": None,
                        "checkpoints": [
                            {
                                "candidate": seed["candidate"],
                                "candidate_hash": seed["candidate_hash"],
                            }
                        ],
                        "analysis": {"terminal_passed": True},
                    }
                ),
                encoding="utf-8",
            )
    return run_dir


def test_compare_accepts_complete_complementary_runs(tmp_path, capsys):
    left = _write_run(tmp_path, "left", ["instant_once"])
    right = _write_run(tmp_path, "right", ["thinking_once"])
    assert (
        compare(argparse.Namespace(run_dir=[str(left), str(right)], output=None)) == 0
    )
    assert "source_run_sha256" in capsys.readouterr().out


def test_compare_rejects_mismatched_fixed_configuration(tmp_path):
    left = _write_run(tmp_path, "left", ["instant_once"])
    right = _write_run(tmp_path, "right", ["thinking_once"])
    manifest_path = right / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["order_seed"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="order_seed"):
        compare(argparse.Namespace(run_dir=[str(left), str(right)], output=None))


def test_compare_rejects_mismatched_revision_protocol(tmp_path):
    left = _write_run(tmp_path, "left", ["instant_once"])
    right = _write_run(tmp_path, "right", ["thinking_once"])
    manifest_path = right / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision_protocol"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="revision_protocol"):
        compare(argparse.Namespace(run_dir=[str(left), str(right)], output=None))


def test_compare_rejects_incomplete_run(tmp_path):
    run_dir = _write_run(
        tmp_path, "incomplete", ["instant_once"], write_results=False
    )
    with pytest.raises(SystemExit, match="Incomplete"):
        compare(argparse.Namespace(run_dir=[str(run_dir)], output=None))


def test_compare_rejects_different_initial_drafts_for_same_reasoning(tmp_path):
    left = _write_run(tmp_path, "left", ["instant_once"])
    right = _write_run(tmp_path, "right", ["instant_recursive"])
    seed_path = right / "seed_drafts.json"
    seeds = json.loads(seed_path.read_text(encoding="utf-8"))
    seeds["records"][0]["candidate"] = "different"
    seeds["records"][0]["candidate_hash"] = hashlib.sha256(
        b"different"
    ).hexdigest()
    seed_path.write_text(json.dumps(seeds), encoding="utf-8")

    with pytest.raises(SystemExit, match="same initial draft"):
        compare(argparse.Namespace(run_dir=[str(left), str(right)], output=None))


def test_reasoning_probe_requires_distinct_telemetry():
    class FakeClient:
        def chat(self, **kwargs):
            tokens = 0 if kwargs["reasoning"] == "off" else 12
            return SimpleNamespace(
                content="323",
                reasoning_hash=None if tokens == 0 else "hash",
                stats={"reasoning_output_tokens": tokens},
                model_instance_id=kwargs["model"],
            )

    _probes, valid = _reasoning_probe(FakeClient(), "instance")
    assert valid is True


def test_client_rejects_non_loopback_lmstudio_endpoint():
    with pytest.raises(InferenceError, match="local-only"):
        _client(
            argparse.Namespace(
                transport="lmstudio",
                base_url="https://api.example.com/v1",
                timeout=1,
                api_token_env=None,
            )
        )


def test_run_rejects_nonpositive_cycle_budget():
    with pytest.raises(SystemExit, match="max-cycles"):
        run(
            argparse.Namespace(
                max_cycles=0,
                max_output_tokens=10,
                revision_max_output_tokens=None,
            )
        )


def test_evolution_journals_preflight_and_includes_it_in_lifecycle_total(
    tmp_path, monkeypatch
):
    benchmark = tmp_path / "cases.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "family": "test",
                "prompt": "Compute 2 + 3.",
                "output_contract": "Return only the integer.",
                "evaluator": {"kind": "exact", "expected": "5"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class EvolutionClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            role = kwargs["role"]
            content = "323" if role == "capability_probe" else "5"
            return Generation(
                role=role,
                content=content,
                reasoning_hash=(
                    "reasoning" if kwargs["reasoning"] == "on" else None
                ),
                stats={
                    "input_tokens": 10,
                    "total_output_tokens": 2,
                    "reasoning_output_tokens": (
                        1 if kwargs["reasoning"] == "on" else 0
                    ),
                    "transport_attempts": 1,
                    "provider_cost": 0,
                },
                latency_seconds=0.25,
                request_hash=f"request-{len(self.calls)}",
                model_instance_id="model",
            )

    client = EvolutionClient()

    def runtime_context(_args, *, probe_reasoning):
        assert probe_reasoning is False
        return {
            "client": client,
            "model_record": {"id": "model"},
            "model_instance_id": "model",
            "model_record_sha256": "record",
            "model_catalog": None,
            "transport": "lmstudio",
            "transport_provenance": {
                "transport": "test",
            },
            "model_dir": None,
            "model_artifact_sha256": None,
            "chat_template": None,
            "chat_template_sha256": None,
            "lmstudio": {"short_version": "test"},
            "source": {"commit": "abc", "dirty": False},
            "capability_probe": {},
            "max_generation_tokens": 1024,
        }

    monkeypatch.setattr("cosmo.cli._runtime_context", runtime_context)
    args = parser().parse_args(
        [
            "evolve",
            "--model",
            "model",
            "--benchmark",
            str(benchmark),
            "--target-correct",
            "1",
            "--max-cycles",
            "1",
            "--search-root",
            str(tmp_path / "searches"),
        ]
    )

    assert evolve(args) == 0
    run_dir = next(
        path
        for path in (tmp_path / "searches").iterdir()
        if path.is_dir()
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    work = summary["search_work"]["inference_journal"]
    assert manifest["capability_probe_validated"] is True
    assert work["accounting_scope"] == (
        "all journaled evolution lifecycle inference"
    )
    assert work["by_role"]["capability_probe"]["completed_operations"] == 2
    assert work["completed_operations"] == 4
    assert work["total_tokens"] == 48
    assert len(client.calls) == 4


def test_run_reuses_one_initial_draft_across_paired_reasoning_group(
    tmp_path, monkeypatch
):
    benchmark = tmp_path / "cases.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "family": "test",
                "prompt": "Compute 2 + 3.",
                "output_contract": "Return only the integer.",
                "evaluator": {"kind": "exact", "expected": "5"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            role = kwargs["role"]
            content = "4" if role == "draft" else "5"
            return Generation(
                role=role,
                content=content,
                reasoning_hash=None,
                stats={
                    "input_tokens": 10,
                    "total_output_tokens": 1,
                    "reasoning_output_tokens": 0,
                    "transport_attempts": 1,
                    "provider_cost": 0,
                },
                latency_seconds=0.1,
                request_hash=f"request-{len(self.calls)}",
                model_instance_id="model",
            )

    client = FakeClient()
    monkeypatch.setattr(
        "cosmo.cli._runtime_context",
        lambda _args: {
            "client": client,
            "model_record": {"id": "model"},
            "model_instance_id": "model",
            "model_record_sha256": "record",
            "transport": "lmstudio",
            "transport_provenance": {"transport": "test"},
            "model_dir": None,
            "model_artifact_sha256": None,
            "chat_template": None,
            "chat_template_sha256": None,
            "lmstudio": {"short_version": "test"},
            "source": {"commit": "abc", "dirty": False},
            "capability_probe": {},
        },
    )
    args = parser().parse_args(
        [
            "run",
            "--model",
            "model",
            "--benchmark",
            str(benchmark),
            "--arms",
            "instant_once,instant_recursive",
            "--max-cycles",
            "2",
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    assert run(args) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    once = json.loads(
        (run_dir / "results" / "case-1--instant_once.json").read_text()
    )
    recursive = json.loads(
        (run_dir / "results" / "case-1--instant_recursive.json").read_text()
    )
    assert once["checkpoints"][0]["candidate"] == "4"
    assert recursive["checkpoints"][0]["candidate"] == "4"
    assert (
        once["generations"][0]["request_hash"]
        == recursive["generations"][0]["request_hash"]
    )
    assert [call["role"] for call in client.calls].count("draft") == 1
    assert (run_dir / "seed_drafts.json").is_file()


def test_chat_input_mode_dispatch_choice():
    """Parser-level mode selection: --message, TTY REPL, or piped stdin."""
    assert resolve_chat_input_mode("List tasks.", stdin_isatty=True) == "message"
    assert resolve_chat_input_mode("List tasks.", stdin_isatty=False) == "message"
    assert resolve_chat_input_mode(None, stdin_isatty=True) == "repl"
    assert resolve_chat_input_mode(None, stdin_isatty=False) == "stdin"
    # Empty --message stays on the one-shot path (rejected later as empty).
    assert resolve_chat_input_mode("", stdin_isatty=True) == "message"
    assert resolve_chat_input_mode("   ", stdin_isatty=False) == "message"


def test_chat_parser_accepts_optional_message():
    args = parser().parse_args(
        ["chat", "--model", "local-model", "--message", "hello"]
    )
    assert args.func.__name__ == "chat_command"
    assert args.model == "local-model"
    assert args.message == "hello"
    args_repl = parser().parse_args(["chat", "--model", "local-model"])
    assert args_repl.message is None
    root_help = parser().format_help().lower()
    assert "control plane" in root_help


def test_control_plan_integration_and_inspect_commands(tmp_path, capsys):
    from cosmo.atomic import AtomicStore
    from cosmo.cli import control_command
    from cosmo.control import ControlPlane

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    state = tmp_path / "state"
    plane = ControlPlane(AtomicStore(state))
    plane.submit_request(
        request_text="x",
        objective="x",
        artifact_path="mod.py",
        repo_root=repo,
        validators=["python_syntax"],
        priority_components={
            "urgency": 0.5,
            "impact": 0.5,
            "freshness": 1.0,
            "operator_boost": 0.0,
            "blocked_penalty": 0.0,
            "dependency_penalty": 0.0,
        },
    )

    # list-plan-versions (empty is fine)
    args = parser().parse_args(
        ["control", "list-plan-versions", "--state-root", str(state)]
    )
    assert control_command(args) == 0
    assert json.loads(capsys.readouterr().out) == []

    # inspect-repo map / list / read
    args = parser().parse_args(
        [
            "control",
            "inspect-repo",
            "--state-root",
            str(state),
            "--repo-root",
            str(repo),
            "--mode",
            "map",
        ]
    )
    assert control_command(args) == 0
    mapped = json.loads(capsys.readouterr().out)
    assert mapped.get("map_sha256") or mapped.get("files") is not None

    args = parser().parse_args(
        [
            "control",
            "inspect-repo",
            "--state-root",
            str(state),
            "--repo-root",
            str(repo),
            "--mode",
            "read",
            "--path",
            "mod.py",
        ]
    )
    assert control_command(args) == 0
    read = json.loads(capsys.readouterr().out)
    assert "VALUE" in read["excerpt"]

    # Unsafe path refused
    args = parser().parse_args(
        [
            "control",
            "inspect-repo",
            "--state-root",
            str(state),
            "--repo-root",
            str(repo),
            "--mode",
            "read",
            "--path",
            "../secret",
        ]
    )
    assert control_command(args) == 2

    # list-integrations empty
    args = parser().parse_args(
        ["control", "list-integrations", "--state-root", str(state)]
    )
    assert control_command(args) == 0
    assert json.loads(capsys.readouterr().out) == []

    # Parser wires assemble/review/promote
    for command in (
        [
            "control",
            "assemble-integration",
            "--repo-root",
            str(repo),
            "--parent-node-id",
            "NODE-1",
        ],
        [
            "control",
            "review-integration",
            "--integration-id",
            "INTEG-1",
            "--verdict",
            "PASS",
        ],
        [
            "control",
            "promote-integration",
            "--integration-id",
            "INTEG-1",
            "--repo-root",
            str(repo),
        ],
        [
            "control",
            "get-plan-version",
            "--plan-version-id",
            "PLANVER-1",
        ],
        [
            "control",
            "get-integration",
            "--integration-id",
            "INTEG-1",
        ],
    ):
        parsed = parser().parse_args(command + ["--state-root", str(state)])
        assert parsed.func.__name__ == "control_command"
