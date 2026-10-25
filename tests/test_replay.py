import json
from pathlib import Path

import pytest

from cosmo.cli import parser
from cosmo.lmstudio import canonical_request_hash, chat_request_body
from cosmo.replay import ReplayClient, ReplayError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "replay" / "trace-accumulator.json"
BENCHMARK = ROOT / "benchmarks" / "iterative-self-verification" / "v1" / "cases.jsonl"


def _chat(client, **overrides):
    request = {
        "model": "replay-model",
        "role": "draft",
        "system_prompt": "system",
        "prompt": "x = (x * 2) - i",
        "reasoning": "off",
        "temperature": 0.0,
        "max_output_tokens": 64,
    }
    request.update(overrides)
    return client.chat(**request)


def test_replay_client_is_deterministic_and_hashes_like_lmstudio():
    client = ReplayClient.from_file(FIXTURE)
    first = _chat(client)
    second = _chat(client)
    assert first.content == second.content == "7"
    expected_hash = canonical_request_hash(
        chat_request_body(
            model="replay-model",
            system_prompt="system",
            prompt="x = (x * 2) - i",
            reasoning="off",
            temperature=0.0,
            max_output_tokens=64,
        )
    )
    assert first.request_hash == expected_hash
    assert first.stats["cosmo_replay"] is True
    assert first.latency_seconds == 0.0


def test_replay_client_refuses_unmatched_requests(tmp_path):
    client = ReplayClient.from_file(FIXTURE)
    with pytest.raises(ReplayError, match="no replay response"):
        _chat(client, prompt="an unrelated task")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")
    with pytest.raises(ReplayError, match="schema_version"):
        ReplayClient.from_file(bad)


def test_cosmo_run_replay_executes_end_to_end_offline(tmp_path, capsys):
    args = parser().parse_args(
        [
            "run",
            "--replay",
            str(FIXTURE),
            "--model",
            "replay-model",
            "--benchmark",
            str(BENCHMARK),
            "--case",
            "trace-accumulator-001",
            "--run-root",
            str(tmp_path),
        ]
    )
    assert args.func(args) == 0

    (run_dir,) = tmp_path.iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["transport"] == "replay"
    assert manifest["transport_provenance"]["evidence"] is False
    assert manifest["base_url"] is None
    assert manifest["capability_probe"]["on"]["stats"]["reasoning_output_tokens"] > 0

    summary = json.loads((run_dir / "summary.json").read_text())
    arms = summary["arms"]
    assert arms["instant_once"]["terminal_successes"] == 0
    assert arms["instant_recursive"]["terminal_successes"] == 1
    assert arms["instant_recursive"]["corrected_runs"] == 1
    assert arms["thinking_once"]["terminal_successes"] == 1
    assert arms["thinking_recursive"]["terminal_successes"] == 1
    # Two shared drafts plus three revisions: 7 -> 5, 5 -> 5, and 5 -> 5.
    assert summary["actual_transport"]["calls"] == 5
    assert len(list((run_dir / "results").glob("*.json"))) == 4
