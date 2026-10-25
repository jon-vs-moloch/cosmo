from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .atomic import (
    AtomicHarness,
    AtomicHarnessError,
    AtomicStaleError,
    AtomicStore,
)
from .authority import (
    approve_repo_root,
    authenticate_model,
    authenticate_operator,
)
from .benchmark import load_cases
from .chat import ChatError, ChatSession, build_system_prompt
from .control import ControlPlane, ControlPlaneError
from .convergence import (
    DRAFT_SYSTEM,
    REVISION_INSTRUCTION,
    REVISION_SYSTEM,
    WITNESSED_REVISION_INSTRUCTION,
    WITNESSED_REVISION_SYSTEM,
    content_hash,
    generate_draft,
    revision_protocol_id,
    run_arm,
    utc_now,
)
from .inference import InferenceClient, InferenceError
from .journal import JournaledInferenceClient, RunDirectoryLock
from .lmstudio import LMStudioClient
from .replay import ReplayClient
from .models import ARMS
from .evolution import (
    BASE_REVISOR_PROMPT,
    EVOLUTION_PROTOCOL,
    MUTATION_KERNEL,
    EvolutionConfig,
    EvolutionSearch,
    EvolutionSearchError,
    EvolutionWriter,
)
from .provenance import (
    git_state,
    lmstudio_app_version,
    model_artifact_hash,
    run_content_hash,
    sha256_file,
)
from .reporting import RunWriter, canonical_hash, summarize, write_json
from .tools import MODEL_REPO_ROOT_TOOLS, ToolDispatcher, tool_definitions_json


def _client(
    args: argparse.Namespace,
) -> InferenceClient:
    if getattr(args, "replay", None):
        return ReplayClient.from_file(args.replay)
    if args.transport != "lmstudio":
        raise InferenceError(
            "Cosmo is local-only; configure the operator-owned LM Studio server."
        )
    token_env = args.api_token_env or "LM_API_TOKEN"
    return LMStudioClient(
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        api_token=os.environ.get(token_env),
    )


def _reasoning_probe(
    client: InferenceClient,
    model_instance_id: str,
    *,
    operation_scope: str | None = None,
) -> tuple[dict[str, object], bool]:
    probes: dict[str, object] = {}
    for reasoning in ("off", "on"):
        generation = client.chat(
            model=model_instance_id,
            role="capability_probe",
            system_prompt="Solve the problem and return only the integer answer.",
            prompt="What is 17 multiplied by 19?",
            reasoning=reasoning,
            temperature=0,
            max_output_tokens=1024,
            operation_id=(
                f"{operation_scope}/{reasoning}"
                if operation_scope is not None
                else None
            ),
        )
        probes[reasoning] = {
            "content_sha256": hashlib.sha256(
                generation.content.encode("utf-8")
            ).hexdigest(),
            "content": generation.content,
            "reasoning_hash": generation.reasoning_hash,
            "stats": generation.stats,
            "model_instance_id": generation.model_instance_id,
        }
    off = probes["off"]
    on = probes["on"]
    valid = (
        int(off["stats"].get("reasoning_output_tokens", 0)) == 0
        and int(on["stats"].get("reasoning_output_tokens", 0)) > 0
        and off["model_instance_id"] in {None, model_instance_id}
        and on["model_instance_id"] in {None, model_instance_id}
    )
    return probes, valid


def _normalize_output_budgets(
    args: argparse.Namespace, *, default: int = 262_144
) -> None:
    """Use the loaded model's advertised ceiling when no cap was requested.

    The experiment measures inference cost; a hidden 1k default can truncate
    otherwise valid model responses and confound accuracy.  The provider still
    enforces its own context/output limits, and runtime callers may pass a
    smaller explicit cap for a controlled cohort.
    """
    if getattr(args, "max_output_tokens", None) is None:
        args.max_output_tokens = default
    if getattr(args, "revision_max_output_tokens", None) is None:
        args.revision_max_output_tokens = args.max_output_tokens


def doctor(args: argparse.Namespace) -> int:
    if args.transport != "lmstudio":
        print(
            "ERROR: Cosmo is local-only; configure the operator-owned LM Studio "
            "server.",
            file=sys.stderr,
        )
        return 2
    try:
        client = _client(args)
        models = client.models()
    except InferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rows = []
    for model in models:
        reasoning = ((model.get("capabilities") or {}).get("reasoning") or {}).get(
            "allowed_options", []
        )
        if reasoning:
            rows.append(
                {
                    "key": model.get("key"),
                    "display_name": model.get("display_name"),
                    "reasoning": reasoning,
                    "loaded_instances": [
                        instance.get("id")
                        for instance in model.get("loaded_instances") or []
                    ],
                }
            )
    output: dict[str, object] = {
        "base_url": args.base_url,
        "reasoning_models": rows,
    }
    if args.probe:
        if not args.model:
            print("ERROR: --probe requires --model", file=sys.stderr)
            return 2
        try:
            model_record = client.require_loaded_reasoning_model(args.model)
            instances = list(model_record.get("loaded_instances") or [])
            instance_id = next(
                (
                    item.get("id")
                    for item in instances
                    if item.get("id") == args.model
                ),
                instances[0].get("id"),
            )
            probes, valid = _reasoning_probe(client, str(instance_id))
            output["probe"] = probes
            output["probe_distinguishes_reasoning"] = valid
        except InferenceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(output, indent=2))
    if args.probe and not output.get("probe_distinguishes_reasoning"):
        return 2
    return 0 if any(row["loaded_instances"] for row in rows) else 1


def _runtime_context(
    args: argparse.Namespace,
    *,
    probe_reasoning: bool = True,
) -> dict[str, object]:
    client = _client(args)
    repo_root = Path(__file__).resolve().parents[2]
    source = git_state(repo_root)
    if isinstance(client, ReplayClient):
        return _replay_runtime_context(
            args, client, source, probe_reasoning=probe_reasoning
        )
    if source["dirty"] and not args.allow_dirty:
        raise InferenceError(
            "Cosmo source tree is dirty. Commit the treatment or pass "
            "--allow-dirty for a non-claim development run."
        )

    if args.transport == "lmstudio":
        if not isinstance(client, LMStudioClient):
            raise InferenceError("LM Studio transport constructed the wrong client")
        model_record = client.require_loaded_reasoning_model(args.model)
        loaded_instances = list(model_record.get("loaded_instances") or [])
        requested_instance = next(
            (item for item in loaded_instances if item.get("id") == args.model),
            None,
        )
        loaded_instance = requested_instance or loaded_instances[0]
        model_instance_id = str(loaded_instance.get("id") or args.model)
        if probe_reasoning:
            capability_probe, probe_valid = _reasoning_probe(
                client, model_instance_id
            )
            if not probe_valid:
                raise InferenceError(
                    "loaded instance did not produce the required off/on reasoning "
                    "telemetry; refusing an invalid cohort"
                )
        else:
            capability_probe = {}
        if not args.model_dir or not args.chat_template:
            raise InferenceError(
                "LM Studio transport requires --model-dir and --chat-template"
            )
        model_dir = Path(args.model_dir).resolve()
        chat_template = Path(args.chat_template).resolve()
        if not model_dir.is_dir() or not chat_template.is_file():
            raise InferenceError("--model-dir and --chat-template must exist")
        print(f"Hashing fixed model artifact: {model_dir}", flush=True)
        loaded_context = int(
            (loaded_instance.get("config") or {}).get("context_length", 0)
        )
        max_context = int(model_record.get("max_context_length") or 0)
        return {
            "transport": args.transport,
            "client": client,
            "model_record": model_record,
            "model_instance_id": model_instance_id,
            "capability_probe": capability_probe,
            "source": source,
            "model_dir": model_dir,
            "model_artifact_sha256": model_artifact_hash(model_dir),
            "chat_template": chat_template,
            "chat_template_sha256": sha256_file(chat_template),
            "model_record_sha256": canonical_hash(model_record),
            "lmstudio": lmstudio_app_version(),
            "transport_provenance": {
                "transport": "lmstudio",
                "base_url": args.base_url,
                "api_token_env": args.api_token_env or "LM_API_TOKEN",
            },
            "model_catalog": None,
            "max_context_length": max_context,
            "max_generation_tokens": loaded_context or max_context,
        }

    raise InferenceError(
        "Cosmo is local-only; no remote inference transport is available."
    )


def _replay_runtime_context(
    args: argparse.Namespace,
    client: ReplayClient,
    source: dict[str, object],
    *,
    probe_reasoning: bool,
) -> dict[str, object]:
    """Runtime for the offline replay transport (`cosmo run --replay`).

    Replay exercises the full harness path without a model server. It applies
    the same loaded-model and reasoning-probe preflight as LM Studio, records
    the fixture hash as transport provenance, and is never claim evidence, so
    the dirty-source refusal (a claim-integrity guard) does not apply.
    """
    model_record = client.require_loaded_reasoning_model(args.model)
    loaded_instances = list(model_record.get("loaded_instances") or [])
    loaded_instance = next(
        (item for item in loaded_instances if item.get("id") == args.model),
        loaded_instances[0],
    )
    model_instance_id = str(loaded_instance.get("id") or args.model)
    capability_probe: dict[str, object] = {}
    if probe_reasoning:
        capability_probe, probe_valid = _reasoning_probe(client, model_instance_id)
        if not probe_valid:
            raise InferenceError(
                "replay fixture did not produce the required off/on reasoning "
                "telemetry for the capability probe"
            )
    loaded_context = int((loaded_instance.get("config") or {}).get("context_length", 0))
    max_context = int(model_record.get("max_context_length") or 0)
    return {
        "transport": "replay",
        "client": client,
        "model_record": model_record,
        "model_instance_id": model_instance_id,
        "capability_probe": capability_probe,
        "source": source,
        "model_dir": None,
        "model_artifact_sha256": None,
        "chat_template": None,
        "chat_template_sha256": None,
        "model_record_sha256": canonical_hash(model_record),
        "lmstudio": None,
        "transport_provenance": {
            "transport": "replay",
            "fixture": Path(args.replay).name,
            "fixture_sha256": client.fixture_sha256,
            "evidence": False,
        },
        "model_catalog": None,
        "max_context_length": max_context,
        "max_generation_tokens": loaded_context or max_context,
    }


def run(args: argparse.Namespace) -> int:
    if args.max_cycles < 1:
        raise SystemExit("--max-cycles must be positive")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise SystemExit("--max-output-tokens must be positive")
    if (
        args.revision_max_output_tokens is not None
        and args.revision_max_output_tokens < 1
    ):
        raise SystemExit("--revision-max-output-tokens must be positive")
    benchmark_path = Path(args.benchmark).resolve()
    cases = load_cases(benchmark_path)
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case.case_id in requested]
        missing = requested - {case.case_id for case in cases}
        if missing:
            raise SystemExit(f"Unknown case IDs: {', '.join(sorted(missing))}")
    if args.limit:
        cases = cases[: args.limit]
    arm_ids = list(ARMS) if args.arms == "all" else args.arms.split(",")
    unknown_arms = set(arm_ids) - set(ARMS)
    if unknown_arms:
        raise SystemExit(f"Unknown arms: {', '.join(sorted(unknown_arms))}")
    try:
        runtime = _runtime_context(args)
    except InferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    client = runtime["client"]
    advertised_output_ceiling = int(runtime.get("max_generation_tokens") or 0)
    _normalize_output_budgets(
        args, default=advertised_output_ceiling or 262_144
    )
    if advertised_output_ceiling and args.max_output_tokens > advertised_output_ceiling:
        raise SystemExit(
            "--max-output-tokens exceeds the loaded model output ceiling "
            f"({args.max_output_tokens} > {advertised_output_ceiling})"
        )
    if advertised_output_ceiling and args.revision_max_output_tokens > advertised_output_ceiling:
        raise SystemExit(
            "--revision-max-output-tokens exceeds the loaded model output ceiling "
            f"({args.revision_max_output_tokens} > {advertised_output_ceiling})"
        )
    model_record = runtime["model_record"]
    model_instance_id = str(runtime["model_instance_id"])
    revision_instruction = (
        REVISION_INSTRUCTION
        if args.revision_mode == "direct"
        else WITNESSED_REVISION_INSTRUCTION
    )
    revision_prompt_path = None
    if args.revision_prompt_file:
        revision_prompt_path = Path(args.revision_prompt_file).resolve()
        revision_instruction = revision_prompt_path.read_text(encoding="utf-8").strip()
        if not revision_instruction:
            raise SystemExit("--revision-prompt-file contains an empty prompt")
    prompt_hashes = {
        "draft": hashlib.sha256(DRAFT_SYSTEM.encode("utf-8")).hexdigest(),
        "revision_system": hashlib.sha256(
            (
                REVISION_SYSTEM
                if args.revision_mode == "direct"
                else WITNESSED_REVISION_SYSTEM
            ).encode("utf-8")
        ).hexdigest(),
        "revision_instruction": hashlib.sha256(
            revision_instruction.encode("utf-8")
        ).hexdigest(),
    }
    evaluator_hash = sha256_file(Path(__file__).with_name("benchmark.py"))

    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_hash = hashlib.sha256(benchmark_bytes).hexdigest()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_root).resolve() / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = Path(args.run_root).resolve() / f"{run_id}-{suffix}"
        suffix += 1
    writer = RunWriter(run_dir)
    manifest = {
        "schema_version": "cosmo.run.v3",
        "cosmo_version": __version__,
        "run_id": run_dir.name,
        "created_at": utc_now(),
        "benchmark": str(benchmark_path),
        "benchmark_sha256": benchmark_hash,
        "case_ids": [case.case_id for case in cases],
        "model": str(model_record.get("id") or model_record.get("key") or args.model),
        "model_instance_id": model_instance_id,
        "model_record": model_record,
        "model_record_sha256": runtime["model_record_sha256"],
        "model_catalog": runtime.get("model_catalog"),
        "transport": runtime["transport"],
        "transport_provenance": runtime["transport_provenance"],
        "transport_provenance_sha256": canonical_hash(
            runtime["transport_provenance"]
        ),
        "model_artifact_dir": (
            str(runtime["model_dir"]) if runtime["model_dir"] else None
        ),
        "model_artifact_sha256": runtime["model_artifact_sha256"],
        "chat_template_path": (
            str(runtime["chat_template"]) if runtime["chat_template"] else None
        ),
        "chat_template_sha256": runtime["chat_template_sha256"],
        "lmstudio": runtime["lmstudio"],
        "source": runtime["source"],
        "revision_mode": args.revision_mode,
        "revision_protocol": revision_protocol_id(args.revision_mode),
        "revision_prompt_path": (
            str(revision_prompt_path) if revision_prompt_path else None
        ),
        "prompt_hashes": prompt_hashes,
        "evaluator_sha256": evaluator_hash,
        "capability_probe": runtime["capability_probe"],
        "capability_probe_validated": runtime.get(
            "capability_probe_validated", True
        ),
        "local_server_logging_policy": (
            "external_to_cosmo; inspect the operator-owned LM Studio runtime"
        ),
        # None for replay runs: no server was contacted.
        "base_url": runtime["transport_provenance"].get("base_url"),
        "arms": arm_ids,
        "max_cycles": args.max_cycles,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "revision_max_output_tokens": (
            args.revision_max_output_tokens or args.max_output_tokens
        ),
        "order_seed": args.order_seed,
        "store": False,
        "hidden_evaluation_is_terminal": True,
        "initial_draft_policy": (
            "one immutable draft per case and reasoning setting, copied into "
            "every selected arm with its original usage and latency"
        ),
    }
    manifest["config_fingerprint"] = canonical_hash(
        {
            key: value
            for key, value in manifest.items()
            if key not in {
                "run_id",
                "created_at",
                "model_record",
                "capability_probe",
            }
        }
    )
    write_json(run_dir / "manifest.json", manifest)

    reasoning_groups = sorted({ARMS[arm_id].reasoning for arm_id in arm_ids})
    seed_schedule = [
        (case, reasoning)
        for case in cases
        for reasoning in reasoning_groups
    ]
    random.Random(args.order_seed).shuffle(seed_schedule)
    initial_generations = {}
    seed_records = []
    try:
        for sequence, (case, reasoning) in enumerate(seed_schedule):
            representative = next(
                ARMS[arm_id]
                for arm_id in arm_ids
                if ARMS[arm_id].reasoning == reasoning
            )
            generation = generate_draft(
                client=client,
                model=model_instance_id,
                case=case,
                arm=representative,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                sequence=sequence,
            )
            initial_generations[(case.case_id, reasoning)] = generation
            record = {
                "case_id": case.case_id,
                "reasoning": reasoning,
                "candidate": generation.content,
                "candidate_hash": content_hash(generation.content),
                "generation": asdict(generation),
            }
            seed_records.append(record)
            writer.event(
                {
                    "type": "shared_initial_draft_completed",
                    "at": utc_now(),
                    **record,
                }
            )
            print(
                f"{case.case_id:28} shared_draft[{reasoning:3}]     READY "
                f"tokens={generation.stats.get('total_output_tokens', 0)}",
                flush=True,
            )
    except InferenceError as exc:
        writer.event(
            {
                "type": "shared_initial_draft_failed",
                "at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"ERROR: initial draft generation failed: {exc}", file=sys.stderr)
        return 2
    write_json(
        run_dir / "seed_drafts.json",
        {
            "schema_version": "cosmo.seed-drafts.v1",
            "records": seed_records,
        },
    )

    schedule = [(case, arm_id) for case in cases for arm_id in arm_ids]
    random.Random(args.order_seed).shuffle(schedule)
    results = []
    for case, arm_id in schedule:
        writer.event(
            {
                "type": "arm_started",
                "at": utc_now(),
                "case_id": case.case_id,
                "arm_id": arm_id,
            }
        )
        result = run_arm(
            client=client,
            model=model_instance_id,
            case=case,
            arm=ARMS[arm_id],
            max_cycles=args.max_cycles,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            revision_max_output_tokens=(
                args.revision_max_output_tokens or args.max_output_tokens
            ),
            revision_instruction=revision_instruction,
            revision_mode=args.revision_mode,
            initial_generation=initial_generations[
                (case.case_id, ARMS[arm_id].reasoning)
            ],
            on_generation=lambda generation, case_id=case.case_id, current_arm=arm_id: writer.event(
                {
                    "type": "inference_completed",
                    "at": utc_now(),
                    "case_id": case_id,
                    "arm_id": current_arm,
                    "generation": asdict(generation),
                }
            ),
        )
        writer.result(result)
        results.append(result)
        writer.event(
            {
                "type": "arm_finished",
                "at": utc_now(),
                "case_id": case.case_id,
                "arm_id": arm_id,
                "stop_reason": result.stop_reason,
                "error": result.error,
                "analysis": result.analysis,
            }
        )
        status = "ERROR" if result.error else (
            "PASS" if result.analysis.get("terminal_passed") else "FAIL"
        )
        print(
            f"{case.case_id:28} {arm_id:20} {status:5} "
            f"calls={result.analysis.get('calls', 0)} stop={result.stop_reason}",
            flush=True,
        )

    summary = summarize(results)
    revision_generations = [
        generation
        for result in results
        for generation in result.generations
        if generation.role != "draft"
    ]
    actual_generations = [
        *(record["generation"] for record in seed_records),
        *(asdict(generation) for generation in revision_generations),
    ]
    summary["actual_transport"] = {
        "calls": len(actual_generations),
        "input_tokens": sum(
            int(item["stats"].get("input_tokens", 0))
            for item in actual_generations
        ),
        "output_tokens": sum(
            int(item["stats"].get("total_output_tokens", 0))
            for item in actual_generations
        ),
        "reasoning_tokens": sum(
            int(item["stats"].get("reasoning_output_tokens", 0))
            for item in actual_generations
        ),
        "cached_input_tokens": sum(
            int(item["stats"].get("cached_input_tokens", 0))
            for item in actual_generations
        ),
        "inference_seconds": round(
            sum(
                float(item.get("latency_seconds", 0.0))
                for item in actual_generations
            ),
            6,
        ),
        "transport_attempts": sum(
            max(1, int(item["stats"].get("transport_attempts", 1)))
            for item in actual_generations
        ),
        "transport_retries": sum(
            max(0, int(item["stats"].get("transport_attempts", 1)) - 1)
            for item in actual_generations
        ),
        "shared_initial_draft_calls": len(seed_records),
        "provider_cost": round(
            sum(
                float(item["stats"]["provider_cost"])
                for item in actual_generations
                if isinstance(
                    item["stats"].get("provider_cost"),
                    (int, float, str),
                )
            ),
            12,
        ),
        "provider_cost_observations": sum(
            "provider_cost" in item["stats"]
            for item in actual_generations
        ),
    }
    summary["actual_transport"]["provider_cost_complete"] = (
        summary["actual_transport"]["provider_cost_observations"]
        == summary["actual_transport"]["calls"]
    )
    summary["run_id"] = run_dir.name
    summary["run_dir"] = str(run_dir)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if any(result.error for result in results) else 0


def evolve(args: argparse.Namespace) -> int:
    positive_fields = (
        "revision_population",
        "revisor_population",
        "mutation_calls_per_generation",
        "revision_mutations_per_call",
        "revisor_mutations_per_call",
        "max_cycles",
        "max_output_tokens",
        "revision_max_output_tokens",
    )
    for field in positive_fields:
        value = getattr(args, field)
        if value is not None and int(value) < 1:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    if not args.no_generation_ceiling and args.generations < 1:
        raise SystemExit("--generations must be positive")
    if args.token_penalty_per_1k_per_case < 0:
        raise SystemExit("--token-penalty-per-1k-per-case cannot be negative")
    if args.meta_token_penalty_per_1k < 0:
        raise SystemExit("--meta-token-penalty-per-1k cannot be negative")
    if args.invalid_mutation_penalty < 0:
        raise SystemExit("--invalid-mutation-penalty cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.plateau_generations < 0:
        raise SystemExit("--plateau-generations cannot be negative")
    benchmark_path = Path(args.benchmark).resolve()
    cases = load_cases(benchmark_path)
    if args.limit:
        cases = cases[: args.limit]
    if args.target_correct is not None and not (
        1 <= args.target_correct <= len(cases)
    ):
        raise SystemExit(
            "--target-correct must be between 1 and the post-limit case count "
            f"({len(cases)})"
        )
    resuming = bool(args.resume)
    if resuming:
        run_dir = Path(args.resume).resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--resume is not a search directory: {run_dir}")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(args.search_root).resolve() / run_id
        suffix = 1
        while run_dir.exists():
            run_dir = Path(args.search_root).resolve() / f"{run_id}-{suffix}"
            suffix += 1
    run_lock = RunDirectoryLock(run_dir)
    try:
        run_lock.acquire()
    except InferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        # Evolutionary inference is retried only at the durable logical-
        # operation boundary. A transport-owned retry would hide a possibly
        # billed attempt beneath one journal receipt.
        runtime = _runtime_context(
            args,
            probe_reasoning=False,
        )
    except InferenceError as exc:
        run_lock.release()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    model_record = runtime["model_record"]
    advertised_output_ceiling = int(runtime["max_generation_tokens"])
    _normalize_output_budgets(
        args, default=advertised_output_ceiling or 262_144
    )
    if advertised_output_ceiling and args.max_output_tokens > advertised_output_ceiling:
        raise SystemExit(
            "--max-output-tokens exceeds the loaded model output ceiling "
            f"({args.max_output_tokens} > {advertised_output_ceiling})"
        )
    if advertised_output_ceiling and args.revision_max_output_tokens > advertised_output_ceiling:
        raise SystemExit(
            "--revision-max-output-tokens exceeds the loaded model output ceiling "
            f"({args.revision_max_output_tokens} > {advertised_output_ceiling})"
        )
    mutation_max_output_tokens = (
        args.mutation_max_output_tokens
        if args.mutation_max_output_tokens is not None
        else advertised_output_ceiling or 262_144
    )
    if mutation_max_output_tokens < 1:
        raise SystemExit(
            "Model does not advertise a usable output ceiling; set "
            "--mutation-max-output-tokens explicitly"
        )
    if (
        advertised_output_ceiling
        and mutation_max_output_tokens > advertised_output_ceiling
    ):
        raise SystemExit(
            "Prompt-revisor output ceiling exceeds the fixed local model "
            f"ceiling ({mutation_max_output_tokens} > "
            f"{advertised_output_ceiling})"
        )

    config = EvolutionConfig(
        generations=None if args.no_generation_ceiling else args.generations,
        revision_population=args.revision_population,
        revisor_population=args.revisor_population,
        mutation_calls_per_generation=args.mutation_calls_per_generation,
        revision_mutations_per_call=args.revision_mutations_per_call,
        revisor_mutations_per_call=args.revisor_mutations_per_call,
        max_cycles=args.max_cycles,
        task_temperature=args.temperature,
        mutation_temperature=args.mutation_temperature,
        draft_max_output_tokens=args.max_output_tokens,
        revision_max_output_tokens=args.revision_max_output_tokens,
        mutation_max_output_tokens=mutation_max_output_tokens,
        revisor_reasoning=args.revisor_reasoning,
        revision_mode=args.revision_mode,
        token_penalty_per_1k_per_case=args.token_penalty_per_1k_per_case,
        meta_token_penalty_per_1k=args.meta_token_penalty_per_1k,
        invalid_mutation_penalty=args.invalid_mutation_penalty,
        order_seed=args.order_seed,
        plateau_generations=args.plateau_generations,
        target_correct=args.target_correct,
        stop_on_collapse=args.stop_on_collapse,
    )
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    evaluator_hash = sha256_file(Path(__file__).with_name("benchmark.py"))
    prompt_hashes = {
        "draft_system": hashlib.sha256(DRAFT_SYSTEM.encode()).hexdigest(),
        "revision_system": hashlib.sha256(
            (
                REVISION_SYSTEM
                if args.revision_mode == "direct"
                else WITNESSED_REVISION_SYSTEM
            ).encode()
        ).hexdigest(),
        "seed_revision": hashlib.sha256(
            (
                REVISION_INSTRUCTION
                if args.revision_mode == "direct"
                else WITNESSED_REVISION_INSTRUCTION
            ).encode()
        ).hexdigest(),
        "seed_revisor": hashlib.sha256(BASE_REVISOR_PROMPT.encode()).hexdigest(),
        "mutation_kernel": hashlib.sha256(MUTATION_KERNEL.encode()).hexdigest(),
    }
    existing_manifest = None
    if resuming:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            run_lock.release()
            raise SystemExit("Resumed search has no manifest.json")
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    manifest = {
        "schema_version": "cosmo.evolution.v2",
        "cosmo_version": __version__,
        "protocol": EVOLUTION_PROTOCOL,
        "run_id": run_dir.name,
        "created_at": utc_now(),
        "benchmark": str(benchmark_path),
        "benchmark_role": "optimization_search_data",
        "benchmark_sha256": benchmark_hash,
        "case_ids": [case.case_id for case in cases],
        "evaluator_sha256": evaluator_hash,
        "model": str(
            model_record.get("id") or model_record.get("key") or args.model
        ),
        "model_instance_id": runtime["model_instance_id"],
        "model_record": runtime["model_record"],
        "model_record_sha256": runtime["model_record_sha256"],
        "model_catalog": runtime.get("model_catalog"),
        "transport": runtime["transport"],
        "transport_provenance": runtime["transport_provenance"],
        "transport_provenance_sha256": canonical_hash(
            runtime["transport_provenance"]
        ),
        "model_artifact_dir": (
            str(runtime["model_dir"]) if runtime["model_dir"] else None
        ),
        "model_artifact_sha256": runtime["model_artifact_sha256"],
        "chat_template_path": (
            str(runtime["chat_template"]) if runtime["chat_template"] else None
        ),
        "chat_template_sha256": runtime["chat_template_sha256"],
        "lmstudio": runtime["lmstudio"],
        "source": runtime["source"],
        "capability_probe": (
            existing_manifest.get("capability_probe")
            if existing_manifest is not None
            else runtime.get("capability_probe", {})
        ),
        "capability_probe_validated": (
            bool(existing_manifest.get("capability_probe_validated"))
            if existing_manifest is not None
            else bool(runtime.get("capability_probe_validated", True))
        ),
        "base_url": runtime["transport_provenance"].get(
            "base_url", args.base_url
        ),
        "store": False,
        "config": asdict(config),
        "prompt_hashes": prompt_hashes,
        "reward": {
            "revision": (
                "lexicographic(valid, terminal_successes, fewer false returns, "
                "fewer regressions/protocol errors, more corrections, fewer "
                "charged tokens)"
            ),
            "reported_revision_fitness": (
                "terminal_accuracy - token_penalty_per_1k_per_case * "
                "(mean_charged_task_tokens_per_case / 1000)"
            ),
            "revisor": (
                "mean requested selection credit; proposal token penalty is "
                "a final tie-break"
            ),
            "revisor_slot_credit": (
                "revision child = accuracy delta, then aggregate verification "
                "behavior delta, then token delta; valid revisor child = 0 "
                "until it acts; invalid or duplicate slot = "
                "-invalid_mutation_penalty"
            ),
            "charged_tokens": "input_tokens + total_output_tokens",
            "fitness_token_scope": (
                "revision evaluation only; the invariant frozen-draft corpus "
                "is reported separately and does not affect ranking"
            ),
            "selection_is_harness_owned": True,
            "accuracy_is_not_projected_to_prompt_revisor": True,
            "selection_credit_is_not_projected_to_prompt_revisor": True,
            "case_outcomes_are_withheld": True,
        },
        "fixed_seed_drafts_reused_across_revision_prompts": True,
        "inference_journal": {
            "schema_version": "cosmo.inference-receipt.v1",
            "stable_operation_ids": True,
            "completed_operations_replayed_without_inference": True,
            "unknown_outcome_attempts_are_reported": True,
            "transport_attempts_per_logical_operation": 1,
            "capability_probes_are_journaled": True,
        },
        "hidden_final_evaluation": None,
    }
    manifest["config_fingerprint"] = canonical_hash(
        {
            key: value
            for key, value in manifest.items()
            if key
            not in {
                "run_id",
                "created_at",
                "model_record",
                "capability_probe",
                "capability_probe_validated",
            }
        }
    )
    if resuming:
        assert existing_manifest is not None
        fixed_resume_fields = (
            "schema_version",
            "cosmo_version",
            "protocol",
            "benchmark_sha256",
            "case_ids",
            "evaluator_sha256",
            "model",
            "model_instance_id",
            "model_record_sha256",
            "transport",
            "transport_provenance_sha256",
            "model_artifact_sha256",
            "chat_template_sha256",
            "lmstudio",
            "source",
            "base_url",
            "store",
            "config",
            "prompt_hashes",
            "reward",
            "inference_journal",
        )
        mismatched = [
            field
            for field in fixed_resume_fields
            if existing_manifest.get(field) != manifest.get(field)
        ]
        if mismatched:
            run_lock.release()
            raise SystemExit(
                "Resumed search does not hold fixed: " + ", ".join(mismatched)
            )
        writer = EvolutionWriter(run_dir, resume=True)
    else:
        writer = EvolutionWriter(run_dir)
        write_json(run_dir / "manifest.json", manifest)
    journaled_client = JournaledInferenceClient(
        runtime["client"],
        run_dir / "inference-journal",
    )
    if runtime.get("reasoning_probe_required", True):
        try:
            capability_probe, probe_valid = _reasoning_probe(
                journaled_client,
                str(runtime["model_instance_id"]),
                operation_scope="preflight/reasoning",
            )
        except InferenceError as exc:
            writer.event(
                {
                    "type": "capability_probe_error",
                    "at": utc_now(),
                    "error": str(exc),
                }
            )
            print(f"ERROR: {exc}", file=sys.stderr)
            run_lock.release()
            return 2
        if not probe_valid:
            writer.event(
                {
                    "type": "capability_probe_invalid",
                    "at": utc_now(),
                }
            )
            print(
                "ERROR: fixed model did not produce distinct off/on reasoning "
                "telemetry; refusing an invalid cohort",
                file=sys.stderr,
            )
            run_lock.release()
            return 2
    else:
        capability_probe = dict(runtime.get("capability_probe") or {})
        probe_valid = False
    if runtime.get("reasoning_probe_required", True) and not manifest[
        "capability_probe_validated"
    ]:
        persisted_manifest = (
            dict(existing_manifest)
            if existing_manifest is not None
            else manifest
        )
        persisted_manifest["capability_probe"] = capability_probe
        persisted_manifest["capability_probe_validated"] = True
        write_json(run_dir / "manifest.json", persisted_manifest)
        manifest = persisted_manifest
    search = EvolutionSearch(
        client=journaled_client,
        model=str(runtime["model_instance_id"]),
        cases=cases,
        config=config,
        writer=writer,
        notify_milestones=args.notify_milestones,
    )
    try:
        summary = search.run(resume=resuming)
    except KeyboardInterrupt:
        writer.event({"type": "search_interrupted", "at": utc_now()})
        print(
            f"Search interrupted; resume from generation checkpoint: {run_dir}",
            file=sys.stderr,
        )
        run_lock.release()
        return 130
    except (InferenceError, EvolutionSearchError) as exc:
        writer.event({"type": "search_error", "at": utc_now(), "error": str(exc)})
        print(f"ERROR: {exc}", file=sys.stderr)
        run_lock.release()
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_lock.release()
    return 0


def compare(args: argparse.Namespace) -> int:
    from .models import ArmResult

    fixed_fields = (
        "schema_version",
        "cosmo_version",
        "benchmark_sha256",
        "case_ids",
        "model",
        "model_instance_id",
        "model_record_sha256",
        "transport",
        "transport_provenance_sha256",
        "model_artifact_sha256",
        "chat_template_sha256",
        "lmstudio",
        "source",
        "revision_mode",
        "revision_protocol",
        "prompt_hashes",
        "evaluator_sha256",
        "capability_probe_validated",
        "base_url",
        "max_cycles",
        "temperature",
        "max_output_tokens",
        "revision_max_output_tokens",
        "order_seed",
        "store",
        "hidden_evaluation_is_terminal",
        "initial_draft_policy",
    )
    manifests = []
    results = []
    seen: set[tuple[str, str]] = set()
    compared_seed_hashes: dict[tuple[str, str], str] = {}
    for raw_dir in args.run_dir:
        run_dir = Path(raw_dir).resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        seeds_path = run_dir / "seed_drafts.json"
        if not seeds_path.is_file():
            raise SystemExit(
                f"Compared run has no enforced seed corpus: {run_dir.name}"
            )
        seed_document = json.loads(seeds_path.read_text(encoding="utf-8"))
        raw_seed_records = seed_document.get("records")
        if not isinstance(raw_seed_records, list):
            raise SystemExit(
                f"Compared run has an invalid seed corpus: {run_dir.name}"
            )
        run_seeds: dict[tuple[str, str], str] = {}
        for record in raw_seed_records:
            if not isinstance(record, dict):
                raise SystemExit(
                    f"Compared run has an invalid seed record: {run_dir.name}"
                )
            identity = (
                str(record.get("case_id")),
                str(record.get("reasoning")),
            )
            candidate_hash = str(record.get("candidate_hash") or "")
            candidate = record.get("candidate")
            recomputed_seed_hash = (
                content_hash(candidate)
                if isinstance(candidate, str)
                else ""
            )
            if (
                not candidate_hash
                or candidate_hash != recomputed_seed_hash
                or identity in run_seeds
            ):
                raise SystemExit(
                    f"Compared run has a duplicate or invalid seed hash: "
                    f"{run_dir.name} {identity}"
                )
            run_seeds[identity] = candidate_hash
            prior_hash = compared_seed_hashes.get(identity)
            if prior_hash is not None and prior_hash != candidate_hash:
                raise SystemExit(
                    "Compared runs do not share the same initial draft for "
                    f"{identity[0]} reasoning={identity[1]}"
                )
            compared_seed_hashes[identity] = candidate_hash
        expected = {
            (str(case_id), str(arm_id))
            for case_id in manifest.get("case_ids") or []
            for arm_id in manifest.get("arms") or []
        }
        actual: set[tuple[str, str]] = set()
        for result_path in sorted((run_dir / "results").glob("*.json")):
            value = json.loads(result_path.read_text(encoding="utf-8"))
            identity = (str(value["case_id"]), str(value["arm_id"]))
            actual.add(identity)
            reasoning = str(value["reasoning"])
            checkpoints = value.get("checkpoints")
            checkpoint_zero = (
                checkpoints[0]
                if isinstance(checkpoints, list) and checkpoints
                else None
            )
            observed_seed_hash = (
                str(checkpoint_zero.get("candidate_hash") or "")
                if isinstance(checkpoint_zero, dict)
                else ""
            )
            checkpoint_candidate = (
                checkpoint_zero.get("candidate")
                if isinstance(checkpoint_zero, dict)
                else None
            )
            recomputed_result_hash = (
                content_hash(checkpoint_candidate)
                if isinstance(checkpoint_candidate, str)
                else ""
            )
            expected_seed_hash = run_seeds.get((identity[0], reasoning))
            if (
                not observed_seed_hash
                or observed_seed_hash != recomputed_result_hash
                or expected_seed_hash is None
                or observed_seed_hash != expected_seed_hash
            ):
                raise SystemExit(
                    "Compared run result does not match its enforced seed: "
                    f"{run_dir.name} {identity[0]} {identity[1]}"
                )
            if identity in seen:
                raise SystemExit(
                    f"Duplicate case/arm across compared runs: {identity[0]} {identity[1]}"
                )
            seen.add(identity)
            results.append(
                ArmResult(
                    case_id=str(value["case_id"]),
                    family=str(value["family"]),
                    arm_id=str(value["arm_id"]),
                    reasoning=reasoning,
                    recursive=bool(value["recursive"]),
                    stop_reason=str(value["stop_reason"]),
                    error=value.get("error"),
                    analysis=dict(value.get("analysis") or {}),
                )
            )
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SystemExit(
                f"Incomplete or inconsistent run {run_dir.name}; "
                f"missing={missing}, extra={extra}"
            )
    baseline = manifests[0]
    for manifest in manifests[1:]:
        mismatched = [
            field for field in fixed_fields if manifest.get(field) != baseline.get(field)
        ]
        if mismatched:
            raise SystemExit(
                "Compared runs do not hold fixed: " + ", ".join(mismatched)
            )
    summary = summarize(results)
    summary["source_runs"] = [manifest["run_id"] for manifest in manifests]
    summary["source_run_sha256"] = {
        manifest["run_id"]: run_content_hash(Path(raw_dir).resolve())
        for manifest, raw_dir in zip(manifests, args.run_dir)
    }
    summary["held_fixed"] = {
        field: baseline.get(field) for field in fixed_fields
    }
    if args.output:
        write_json(Path(args.output).resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def seal(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "manifest.json").is_file() or not (run_dir / "summary.json").is_file():
        raise SystemExit(f"Run is incomplete and cannot be sealed: {run_dir}")
    source_hash = run_content_hash(run_dir)
    destination_root = Path(args.destination).resolve()
    destination = destination_root / run_dir.name
    if destination.exists():
        raise SystemExit(f"Seal destination already exists: {destination}")
    destination_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, destination)
    copied_hash = run_content_hash(destination)
    if copied_hash != source_hash:
        raise SystemExit("Sealed run hash mismatch after copy")
    receipt = {
        "run_id": run_dir.name,
        "source": str(run_dir),
        "sealed_at": utc_now(),
        "sha256": source_hash,
        "destination": str(destination),
    }
    write_json(destination_root / f"{run_dir.name}.seal.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _atomic_runtime(args: argparse.Namespace) -> dict[str, object]:
    """Reuse the local-only LM Studio preflight for atomic Workers/Reviewers."""
    runtime_args = argparse.Namespace(
        transport="lmstudio",
        base_url=args.base_url,
        api_token_env=args.api_token_env,
        timeout=args.timeout,
        allow_dirty=args.allow_dirty,
        model=args.model,
        model_dir=args.model_dir,
        chat_template=args.chat_template,
    )
    return _runtime_context(runtime_args)


def atomic_submit(args: argparse.Namespace) -> int:
    try:
        harness = AtomicHarness(AtomicStore(Path(args.state_root)))
        result = harness.submit(
            request_text=args.request_text,
            objective=args.objective,
            artifact_path=args.artifact,
            repo_root=Path(args.repo_root),
            validators=args.validator,
            desired_outcome=args.desired_outcome,
            explicit_constraints=args.constraint,
        )
    except AtomicHarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def atomic_run(args: argparse.Namespace) -> int:
    retry_contract = None
    retry_fields = {
        "addresses": args.addresses,
        "preserves": args.preserves,
        "differs_from": args.differs_from,
        "expected_improvement": args.expected_improvement,
    }
    if any(retry_fields.values()):
        retry_contract = retry_fields
    try:
        runtime = _atomic_runtime(args)
        state_root = Path(args.state_root).resolve()
        journaled_client = JournaledInferenceClient(
            runtime["client"], state_root / "inference-journal"
        )
        result = AtomicHarness(AtomicStore(state_root)).run_attempt(
            task_id=args.task,
            repo_root=Path(args.repo_root).resolve(),
            client=journaled_client,
            model=str(runtime["model_instance_id"]),
            reasoning=args.reasoning,
            temperature=args.temperature,
            worker_max_output_tokens=args.worker_max_output_tokens,
            review_max_output_tokens=args.review_max_output_tokens,
            strategy_summary=args.strategy_summary,
            retry_contract=retry_contract,
        )
    except (AtomicHarnessError, InferenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["promotable"] else 1


def atomic_promote(args: argparse.Namespace) -> int:
    try:
        result = AtomicHarness(AtomicStore(Path(args.state_root))).promote(
            attempt_id=args.attempt,
            repo_root=Path(args.repo_root).resolve(),
        )
    except AtomicStaleError as exc:
        print(f"STALE: {exc}", file=sys.stderr)
        return 3
    except AtomicHarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def atomic_show(args: argparse.Namespace) -> int:
    try:
        result = AtomicHarness(AtomicStore(Path(args.state_root))).show(args.identifier)
    except AtomicHarnessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def resolve_chat_input_mode(
    message: str | None,
    *,
    stdin_isatty: bool,
) -> str:
    """Choose chat input mode without performing I/O.

    Returns one of: ``message`` (one-shot --message), ``repl`` (interactive
    terminal), or ``stdin`` (piped non-TTY content as one message).
    """
    if message is not None and str(message).strip():
        return "message"
    if message is not None and not str(message).strip():
        # Explicit empty --message is invalid; still one-shot path for errors.
        return "message"
    if stdin_isatty:
        return "repl"
    return "stdin"


def chat_command(args: argparse.Namespace) -> int:
    """Local LM Studio chat with control-plane tools (ephemeral transcript).

    Primary terminal chat: interactive REPL when stdin is a TTY and no
    --message is supplied. One-shot --message and non-TTY pipe remain supported.
    Transcript is not durable; tool events and tasks use AtomicStore as usual.
    """
    if args.transport != "lmstudio":
        print(
            "ERROR: Cosmo is local-only; configure the operator-owned LM Studio "
            "server.",
            file=sys.stderr,
        )
        return 2
    state_root = Path(args.state_root).resolve()
    plane = ControlPlane(AtomicStore(state_root))
    base_client: InferenceClient = _client(args)
    if args.journal:
        client: InferenceClient = JournaledInferenceClient(
            base_client, state_root / "chat-inference-journal"
        )
    else:
        client = base_client
    operator = authenticate_operator(
        "local-cli-operator", authenticated_by="cosmo.chat-cli"
    )
    model_caller = authenticate_model(
        f"lmstudio:{args.model}", authenticated_by="cosmo.chat-cli"
    )
    repo_capability = None
    if args.repo_root:
        repo_capability = approve_repo_root(
            operator,
            repo_root=Path(args.repo_root),
            grantee=model_caller,
            allowed_tools=MODEL_REPO_ROOT_TOOLS,
        )
    dispatcher = ToolDispatcher(
        plane,
        client=client,
        model=args.model,
        caller=model_caller,
        repo_capability=repo_capability,
    )
    session = ChatSession(
        client=client,
        dispatcher=dispatcher,
        model=args.model,
        system_prompt=build_system_prompt(args.system_extra),
        max_tool_turns=args.max_tool_turns,
        reasoning=args.reasoning,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    mode = resolve_chat_input_mode(
        args.message,
        stdin_isatty=sys.stdin.isatty(),
    )
    if mode == "repl":
        return _chat_repl(session, json_output=bool(args.json))
    if mode == "stdin":
        try:
            message = sys.stdin.read()
        except KeyboardInterrupt:
            return 130
    else:
        message = args.message
    if not message or not str(message).strip():
        print(
            "ERROR: provide --message, pipe a user message on stdin, or run "
            "interactively on a TTY",
            file=sys.stderr,
        )
        return 2
    try:
        result = session.handle_user_message(str(message))
    except (ChatError, InferenceError, ControlPlaneError, AtomicHarnessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(result.final_text)
    return 0 if result.stop_reason == "final" else 1


def _chat_repl(session: ChatSession, *, json_output: bool = False) -> int:
    """Simple multi-turn REPL over one ChatSession until EOF or /exit."""
    print(
        "Cosmo chat (local). Type a message, or /exit to quit. "
        "Transcript is ephemeral.",
        file=sys.stderr,
    )
    last_code = 0
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130
        if line == "":
            # EOF
            return last_code
        message = line.strip()
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return last_code
        try:
            result = session.handle_user_message(message)
        except (ChatError, InferenceError, ControlPlaneError, AtomicHarnessError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            last_code = 2
            continue
        if json_output:
            print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
        else:
            print(result.final_text)
        last_code = 0 if result.stop_reason == "final" else 1


def control_command(args: argparse.Namespace) -> int:
    """Operator-facing control-plane inspection and lifecycle helpers."""
    state_root = Path(args.state_root).resolve()
    plane = ControlPlane(AtomicStore(state_root))
    operator_caller = authenticate_operator(
        "local-cli-operator", authenticated_by="cosmo.control-cli"
    )
    action = args.control_command
    try:
        if action == "tools":
            result: object = {
                "model": tool_definitions_json(catalog="model"),
                "operator": tool_definitions_json(catalog="operator"),
            }
        elif action == "ready-queue":
            result = plane.ready_queue()
        elif action == "list-tasks":
            result = plane.list_tasks()
        elif action == "get-task":
            result = plane.get_task(args.task_id)
        elif action == "show":
            # Prefer harness show for mixed identifiers; fall back to control plane.
            try:
                result = plane.harness.show(args.identifier)
            except AtomicHarnessError:
                # Best-effort: try common control accessors.
                identifier = args.identifier
                for reader in (
                    plane.get_task,
                    plane.get_request,
                    plane.get_task_spec,
                    plane.get_plan,
                    plane.get_attempt,
                    plane.get_review,
                    plane.get_lease,
                    plane.get_integration,
                ):
                    try:
                        result = reader(identifier)
                        break
                    except (ControlPlaneError, AtomicHarnessError, OSError, KeyError):
                        continue
                else:
                    from .planning import PlanningService

                    try:
                        result = PlanningService(plane).get_plan_version(identifier)
                    except (ControlPlaneError, AtomicHarnessError, OSError, KeyError):
                        raise AtomicHarnessError(f"record not found: {identifier}")
        elif action == "snapshot":
            result = plane.to_json_dict()
        elif action == "submit":
            result = plane.submit_request(
                request_text=args.request_text,
                objective=args.objective,
                artifact_path=args.artifact,
                repo_root=Path(args.repo_root),
                validators=args.validator or (),
                desired_outcome=args.desired_outcome,
                explicit_constraints=args.constraint or (),
                operator_caller=operator_caller,
            )
        elif action == "ingest-request":
            result = plane.ingest_request(
                request_text=args.request_text,
                desired_outcome=getattr(args, "desired_outcome", None),
                explicit_constraints=getattr(args, "constraint", None) or (),
                requested_priority=getattr(args, "requested_priority", None),
                summary=getattr(args, "summary", None),
            )
        elif action == "propose-triage":
            from .triage import TriageService

            proposal = None
            if getattr(args, "proposal_json", None):
                proposal = json.loads(args.proposal_json)
            result = TriageService(plane).propose_triage(
                args.request_id,
                proposal=proposal,
                operator_caller=operator_caller if proposal is not None else None,
            )
        elif action == "review-triage":
            from .triage import TriageService

            review = None
            if getattr(args, "review_json", None):
                review = json.loads(args.review_json)
            else:
                review = {
                    "verdict": args.verdict,
                    "evidence": list(getattr(args, "evidence", None) or ["operator review"]),
                    "notes": getattr(args, "notes", None) or "",
                    "independent_reviewer": args.actor or "operator",
                }
            result = TriageService(plane).review_triage(
                args.proposal_id,
                review=review,
                independent_reviewer=args.actor or "operator",
                operator_caller=operator_caller,
            )
        elif action == "apply-triage":
            from .triage import TriageService

            result = TriageService(plane).apply_triage(
                args.proposal_id,
                repo_root=Path(args.repo_root) if getattr(args, "repo_root", None) else None,
                validators=getattr(args, "validator", None) or (),
            )
        elif action == "list-triage-proposals":
            from .triage import TriageService

            result = TriageService(plane).list_proposals()
        elif action == "scheduler-tick":
            result = plane.scheduler_tick(
                max_concurrent=getattr(args, "max_concurrent", 2) or 2,
                limit=getattr(args, "limit", None),
            )
        elif action == "reassess-priority":
            components = json.loads(args.priority_json)
            if not isinstance(components, dict):
                raise ValueError("--priority-json must be a JSON object")
            result = plane.reassess_priority(args.task_id, components)
        elif action == "refresh-dependency-readiness":
            result = plane.refresh_dependency_readiness(args.task_id)
        elif action == "inspect-lifecycle":
            result = plane.inspect_lifecycle(getattr(args, "task_id", None))
        elif action == "transition":
            result = plane.transition_task(
                args.task_id,
                args.status,
                reason=args.reason,
                actor=args.actor or "operator",
            )
        elif action == "list-plan-versions":
            from .planning import PlanningService

            result = PlanningService(plane).list_plan_versions()
        elif action == "get-plan-version":
            from .planning import PlanningService

            result = PlanningService(plane).get_plan_version(args.plan_version_id)
        elif action == "list-integrations":
            result = plane.list_integrations()
        elif action == "get-integration":
            result = plane.get_integration(args.integration_id)
        elif action == "assemble-integration":
            selected = None
            if getattr(args, "selected_json", None):
                selected = json.loads(args.selected_json)
                if not isinstance(selected, dict):
                    raise ValueError("--selected-json must be a JSON object")
            result = plane.assemble_integration(
                args.parent_node_id,
                repo_root=Path(args.repo_root),
                selected=selected,
                expected_baseline_tree_sha256=getattr(
                    args, "expected_baseline_tree_sha256", None
                ),
            )
        elif action == "review-integration":
            result = plane.review_integration(
                args.integration_id,
                verdict=args.verdict,
                notes=getattr(args, "notes", None),
                operator_caller=operator_caller,
            )
        elif action == "promote-integration":
            result = plane.promote_integration(
                args.integration_id,
                repo_root=Path(args.repo_root) if args.repo_root else None,
            )
        elif action == "inspect-repo":
            from .context import RepositoryInspector

            repo_root = Path(args.repo_root).resolve()
            if not repo_root.is_dir():
                raise ValueError(f"repo_root is not a directory: {repo_root}")
            # Refuse paths that escape via resolution tricks when operators pass
            # a path that is actually the state root.
            if repo_root == state_root or state_root in repo_root.parents:
                # state under repo is fine; repo that *is* state is not a product root.
                if repo_root == state_root:
                    raise ValueError("refusing to inspect AtomicStore state root")
            inspector = RepositoryInspector(repo_root, state_root=state_root)
            mode = args.inspect_mode
            if mode == "map":
                result = inspector.build_map().to_dict()
            elif mode == "list":
                result = inspector.list_tree(
                    prefix=args.prefix or "",
                    max_entries=args.max_entries,
                )
            elif mode == "read":
                if not args.path:
                    raise ValueError("inspect-repo read requires --path")
                result = inspector.read_artifact(
                    args.path,
                    max_bytes=args.max_bytes,
                )
            elif mode == "search":
                if not args.query:
                    raise ValueError("inspect-repo search requires --query")
                result = inspector.search_text(
                    args.query,
                    prefix=args.prefix or "",
                    max_matches=args.max_matches,
                )
            elif mode == "symbols":
                if not args.path:
                    raise ValueError("inspect-repo symbols requires --path")
                result = inspector.inspect_python_symbols(args.path)
            else:
                raise ValueError(f"unknown inspect mode: {mode}")
        else:
            print(f"ERROR: unknown control action: {action}", file=sys.stderr)
            return 2
    except (ControlPlaneError, AtomicHarnessError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_transport_arguments(
    command: argparse.ArgumentParser, *, timeout: float
) -> None:
    command.add_argument(
        "--transport",
        choices=("lmstudio",),
        default="lmstudio",
        help="Local inference boundary. Cosmo does not call remote model APIs.",
    )
    command.add_argument(
        "--base-url",
        default="http://127.0.0.1:1234",
        help="LM Studio server base URL.",
    )
    command.add_argument(
        "--api-token-env",
        help=(
            "Optional environment-variable name for LM Studio authentication; "
            "defaults to LM_API_TOKEN."
        ),
    )
    command.add_argument("--timeout", type=float, default=timeout)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="cosmo")
    root.add_argument("--version", action="version", version=__version__)
    subparsers = root.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Inspect the local LM Studio capability"
    )
    _add_transport_arguments(doctor_parser, timeout=10.0)
    doctor_parser.add_argument("--model")
    doctor_parser.add_argument("--probe", action="store_true")
    doctor_parser.set_defaults(func=doctor)

    run_parser = subparsers.add_parser("run", help="Run a paired experiment")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--benchmark", required=True)
    run_parser.add_argument("--arms", default="all")
    run_parser.add_argument("--case", action="append")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--max-cycles", type=int, default=4)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-output-tokens", type=int)
    run_parser.add_argument("--revision-max-output-tokens", type=int)
    run_parser.add_argument("--revision-prompt-file")
    run_parser.add_argument(
        "--revision-mode",
        choices=("direct", "witnessed"),
        default="direct",
    )
    run_parser.add_argument("--order-seed", type=int, default=20260724)
    run_parser.add_argument("--run-root", default="runs")
    _add_transport_arguments(run_parser, timeout=300.0)
    run_parser.add_argument(
        "--replay",
        metavar="FIXTURE",
        help=(
            "Answer every inference from a cosmo.replay-fixture.v1 JSON file "
            "instead of LM Studio (offline smoke runs; never evidence)."
        ),
    )
    run_parser.add_argument("--model-dir")
    run_parser.add_argument("--chat-template")
    run_parser.add_argument("--allow-dirty", action="store_true")
    run_parser.set_defaults(func=run)

    evolve_parser = subparsers.add_parser(
        "evolve", help="Evolve revision and prompt-revisor prompts"
    )
    evolve_parser.add_argument("--model", required=True)
    evolve_parser.add_argument("--benchmark", required=True)
    evolve_parser.add_argument("--limit", type=int)
    ceiling_group = evolve_parser.add_mutually_exclusive_group()
    ceiling_group.add_argument("--generations", type=int, default=8)
    ceiling_group.add_argument(
        "--no-generation-ceiling",
        action="store_true",
        help="Run until a target, configured stop, fatal error, or interruption",
    )
    evolve_parser.add_argument("--revision-population", type=int, default=4)
    evolve_parser.add_argument("--revisor-population", type=int, default=3)
    evolve_parser.add_argument(
        "--mutation-calls-per-generation", type=int, default=3
    )
    evolve_parser.add_argument("--revision-mutations-per-call", type=int, default=2)
    evolve_parser.add_argument("--revisor-mutations-per-call", type=int, default=1)
    evolve_parser.add_argument("--max-cycles", type=int, default=4)
    evolve_parser.add_argument("--temperature", type=float, default=0.0)
    evolve_parser.add_argument("--mutation-temperature", type=float, default=0.7)
    evolve_parser.add_argument("--max-output-tokens", type=int)
    evolve_parser.add_argument("--revision-max-output-tokens", type=int)
    evolve_parser.add_argument(
        "--mutation-max-output-tokens",
        type=int,
        help=(
            "Prompt-revisor completion reservation. Defaults to the local "
            "model's advertised output ceiling."
        ),
    )
    evolve_parser.add_argument(
        "--revisor-reasoning", choices=("off", "on"), default="on"
    )
    evolve_parser.add_argument(
        "--revision-mode",
        choices=("direct", "witnessed"),
        default="direct",
    )
    evolve_parser.add_argument(
        "--token-penalty-per-1k-per-case", type=float, default=0.05
    )
    evolve_parser.add_argument(
        "--meta-token-penalty-per-1k", type=float, default=0.01
    )
    evolve_parser.add_argument("--invalid-mutation-penalty", type=float, default=0.05)
    evolve_parser.add_argument("--plateau-generations", type=int, default=0)
    evolve_parser.add_argument(
        "--target-correct",
        type=int,
        help="Stop when any fully evaluated revision prompt reaches N correct cases",
    )
    evolve_parser.add_argument(
        "--notify-milestones",
        action="store_true",
        help="Send at-most-once macOS notifications for new raw-correctness highs",
    )
    evolve_parser.add_argument(
        "--stop-on-collapse",
        action="store_true",
        help="Stop an unbounded search if a generation creates no novel revisions",
    )
    evolve_parser.add_argument("--order-seed", type=int, default=20260724)
    evolve_parser.add_argument("--search-root", default="searches")
    evolve_parser.add_argument("--resume")
    _add_transport_arguments(evolve_parser, timeout=300.0)
    evolve_parser.add_argument("--model-dir")
    evolve_parser.add_argument("--chat-template")
    evolve_parser.add_argument("--allow-dirty", action="store_true")
    evolve_parser.set_defaults(func=evolve)

    atomic_parser = subparsers.add_parser(
        "atomic", help="Run the artifact-centric Cosmo Slice 1 harness"
    )
    atomic_subparsers = atomic_parser.add_subparsers(dest="atomic_command", required=True)

    atomic_submit_parser = atomic_subparsers.add_parser(
        "submit", help="Create an immutable Request and approved leaf TaskSpec"
    )
    atomic_submit_parser.add_argument("--state-root", default=".cosmo")
    atomic_submit_parser.add_argument("--repo-root", required=True)
    atomic_submit_parser.add_argument("--request-text", required=True)
    atomic_submit_parser.add_argument("--desired-outcome")
    atomic_submit_parser.add_argument("--objective", required=True)
    atomic_submit_parser.add_argument("--artifact", required=True)
    atomic_submit_parser.add_argument("--validator", action="append", default=[])
    atomic_submit_parser.add_argument("--constraint", action="append", default=[])
    atomic_submit_parser.set_defaults(func=atomic_submit)

    atomic_run_parser = atomic_subparsers.add_parser(
        "run", help="Execute one isolated Worker and independent Review"
    )
    atomic_run_parser.add_argument("--state-root", default=".cosmo")
    atomic_run_parser.add_argument("--repo-root", required=True)
    atomic_run_parser.add_argument("--task", required=True)
    atomic_run_parser.add_argument("--model", required=True)
    atomic_run_parser.add_argument("--model-dir", required=True)
    atomic_run_parser.add_argument("--chat-template", required=True)
    atomic_run_parser.add_argument("--reasoning", choices=("off", "on"), default="off")
    atomic_run_parser.add_argument("--temperature", type=float, default=0.0)
    atomic_run_parser.add_argument("--worker-max-output-tokens", type=int, default=4096)
    atomic_run_parser.add_argument("--review-max-output-tokens", type=int, default=2048)
    atomic_run_parser.add_argument("--strategy-summary", required=True)
    atomic_run_parser.add_argument("--addresses", action="append", default=[])
    atomic_run_parser.add_argument("--preserves", action="append", default=[])
    atomic_run_parser.add_argument("--differs-from", action="append", default=[])
    atomic_run_parser.add_argument("--expected-improvement", action="append", default=[])
    atomic_run_parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    atomic_run_parser.add_argument("--api-token-env")
    atomic_run_parser.add_argument("--timeout", type=float, default=300.0)
    atomic_run_parser.add_argument("--allow-dirty", action="store_true")
    atomic_run_parser.set_defaults(func=atomic_run)

    atomic_promote_parser = atomic_subparsers.add_parser(
        "promote", help="Atomically promote a reviewed artifact into product state"
    )
    atomic_promote_parser.add_argument("--state-root", default=".cosmo")
    atomic_promote_parser.add_argument("--repo-root", required=True)
    atomic_promote_parser.add_argument("--attempt", required=True)
    atomic_promote_parser.set_defaults(func=atomic_promote)

    atomic_show_parser = atomic_subparsers.add_parser(
        "show", help="Show an atomic Request, Task, Attempt, or Promotion record"
    )
    atomic_show_parser.add_argument("--state-root", default=".cosmo")
    atomic_show_parser.add_argument("identifier")
    atomic_show_parser.set_defaults(func=atomic_show)

    compare_parser = subparsers.add_parser(
        "compare", help="Combine complementary arms from compatible runs"
    )
    compare_parser.add_argument("run_dir", nargs="+")
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=compare)

    seal_parser = subparsers.add_parser(
        "seal", help="Copy a complete raw run into durable evidence"
    )
    seal_parser.add_argument("run_dir")
    seal_parser.add_argument("--destination", default="evidence/runs")
    seal_parser.set_defaults(func=seal)

    seal_evolution_parser = subparsers.add_parser(
        "seal-evolution", help="Copy an evolutionary search into durable evidence"
    )
    seal_evolution_parser.add_argument("run_dir")
    seal_evolution_parser.add_argument(
        "--destination", default="evidence/searches"
    )
    seal_evolution_parser.set_defaults(func=seal)

    chat_parser = subparsers.add_parser(
        "chat",
        help=(
            "Primary local chat control plane over strict tools "
            "(planning, atomic attempts, parent integration; LM Studio only; "
            "TTY REPL, --message one-shot, or stdin pipe; ephemeral transcript)"
        ),
    )
    chat_parser.add_argument("--state-root", default=".cosmo")
    chat_parser.add_argument("--model", required=True)
    chat_parser.add_argument(
        "--repo-root",
        help=(
            "Repository root approved by the local operator for this chat "
            "session. The resolved root is carried out-of-band as an immutable "
            "capability and is never accepted from model tool arguments."
        ),
    )
    chat_parser.add_argument(
        "--message",
        help=(
            "One-shot user message (exit after one turn). "
            "If omitted: interactive REPL on a TTY, else read one message from stdin."
        ),
    )
    chat_parser.add_argument(
        "--max-tool-turns",
        type=int,
        default=8,
        help="Bound the tool-call loop before stopping",
    )
    chat_parser.add_argument("--reasoning", choices=("off", "on"), default="off")
    chat_parser.add_argument("--temperature", type=float, default=0.0)
    chat_parser.add_argument("--max-output-tokens", type=int, default=2048)
    chat_parser.add_argument(
        "--system-extra",
        help="Optional operator context appended to the compact system prompt",
    )
    chat_parser.add_argument(
        "--journal",
        action="store_true",
        help="Journal model calls under state-root/chat-inference-journal",
    )
    chat_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured ChatSessionResult JSON instead of plain final text",
    )
    _add_transport_arguments(chat_parser, timeout=300.0)
    chat_parser.set_defaults(func=chat_command)

    control_parser = subparsers.add_parser(
        "control",
        help="Inspect and drive the durable control plane (no model required)",
    )
    control_sub = control_parser.add_subparsers(
        dest="control_command", required=True
    )
    control_tools = control_sub.add_parser(
        "tools", help="Print JSON-serializable tool definitions"
    )
    control_tools.add_argument("--state-root", default=".cosmo")
    control_tools.set_defaults(func=control_command)

    control_ready = control_sub.add_parser(
        "ready-queue", help="List ready leaf tasks by priority"
    )
    control_ready.add_argument("--state-root", default=".cosmo")
    control_ready.set_defaults(func=control_command)

    control_list = control_sub.add_parser("list-tasks", help="List all tasks")
    control_list.add_argument("--state-root", default=".cosmo")
    control_list.set_defaults(func=control_command)

    control_get = control_sub.add_parser("get-task", help="Get one task by id")
    control_get.add_argument("--state-root", default=".cosmo")
    control_get.add_argument("--task-id", required=True)
    control_get.set_defaults(func=control_command)

    control_show = control_sub.add_parser(
        "show", help="Show a control/atomic record by identifier"
    )
    control_show.add_argument("--state-root", default=".cosmo")
    control_show.add_argument("identifier")
    control_show.set_defaults(func=control_command)

    control_snap = control_sub.add_parser(
        "snapshot", help="JSON snapshot of control-plane projections"
    )
    control_snap.add_argument("--state-root", default=".cosmo")
    control_snap.set_defaults(func=control_command)

    control_submit = control_sub.add_parser(
        "submit",
        help="Leaf/parent shortcut: submit request through ControlPlane",
    )
    control_submit.add_argument("--state-root", default=".cosmo")
    control_submit.add_argument("--repo-root", required=True)
    control_submit.add_argument("--request-text", required=True)
    control_submit.add_argument("--objective", required=True)
    control_submit.add_argument("--artifact", required=True)
    control_submit.add_argument("--desired-outcome")
    control_submit.add_argument("--validator", action="append", default=[])
    control_submit.add_argument("--constraint", action="append", default=[])
    control_submit.set_defaults(func=control_command)

    control_ingest = control_sub.add_parser(
        "ingest-request",
        help="Ingest immutable Request only (triage front door; no Task)",
    )
    control_ingest.add_argument("--state-root", default=".cosmo")
    control_ingest.add_argument("--request-text", required=True)
    control_ingest.add_argument("--desired-outcome")
    control_ingest.add_argument("--constraint", action="append", default=[])
    control_ingest.add_argument("--requested-priority")
    control_ingest.add_argument("--summary")
    control_ingest.set_defaults(func=control_command)

    control_propose_triage = control_sub.add_parser(
        "propose-triage",
        help="Store a triage proposal (pass --proposal-json for operator mode)",
    )
    control_propose_triage.add_argument("--state-root", default=".cosmo")
    control_propose_triage.add_argument("--request-id", required=True)
    control_propose_triage.add_argument(
        "--proposal-json",
        help="Operator-supplied triage proposal JSON object",
    )
    control_propose_triage.set_defaults(func=control_command)

    control_review_triage = control_sub.add_parser(
        "review-triage",
        help="Independent triage review (does not apply)",
    )
    control_review_triage.add_argument("--state-root", default=".cosmo")
    control_review_triage.add_argument("--proposal-id", required=True)
    control_review_triage.add_argument(
        "--verdict",
        default="PASS",
        choices=(
            "PASS",
            "RETRY_TRIAGE",
            "NEEDS_USER_CLARIFICATION",
            "ESCALATE",
        ),
    )
    control_review_triage.add_argument("--notes")
    control_review_triage.add_argument("--evidence", action="append", default=[])
    control_review_triage.add_argument("--actor", default="operator")
    control_review_triage.add_argument(
        "--review-json", help="Full operator review JSON (overrides flags)"
    )
    control_review_triage.set_defaults(func=control_command)

    control_apply_triage = control_sub.add_parser(
        "apply-triage",
        help="Apply approved triage proposal (create/link Task; no product write)",
    )
    control_apply_triage.add_argument("--state-root", default=".cosmo")
    control_apply_triage.add_argument("--proposal-id", required=True)
    control_apply_triage.add_argument("--repo-root")
    control_apply_triage.add_argument("--validator", action="append", default=[])
    control_apply_triage.set_defaults(func=control_command)

    control_list_triage = control_sub.add_parser(
        "list-triage-proposals", help="List triage proposals"
    )
    control_list_triage.add_argument("--state-root", default=".cosmo")
    control_list_triage.set_defaults(func=control_command)

    control_sched = control_sub.add_parser(
        "scheduler-tick",
        help="Deterministic ready-queue selection under concurrency limits",
    )
    control_sched.add_argument("--state-root", default=".cosmo")
    control_sched.add_argument("--max-concurrent", type=int, default=2)
    control_sched.add_argument("--limit", type=int)
    control_sched.set_defaults(func=control_command)

    control_reassess = control_sub.add_parser(
        "reassess-priority",
        help="Append a new explainable priority assessment for a task",
    )
    control_reassess.add_argument("--state-root", default=".cosmo")
    control_reassess.add_argument("--task-id", required=True)
    control_reassess.add_argument(
        "--priority-json",
        required=True,
        help="JSON object of explainable priority components",
    )
    control_reassess.set_defaults(func=control_command)

    control_refresh_deps = control_sub.add_parser(
        "refresh-dependency-readiness",
        help="Refresh blocked/ready status from plan dependency satisfaction",
    )
    control_refresh_deps.add_argument("--state-root", default=".cosmo")
    control_refresh_deps.add_argument("--task-id", required=True)
    control_refresh_deps.set_defaults(func=control_command)

    control_lifecycle = control_sub.add_parser(
        "inspect-lifecycle",
        help="Inspect task lifecycle or control-plane snapshot (read-only)",
    )
    control_lifecycle.add_argument("--state-root", default=".cosmo")
    control_lifecycle.add_argument(
        "--task-id",
        help="Optional task id; omit for full control-plane snapshot",
    )
    control_lifecycle.set_defaults(func=control_command)

    control_transition = control_sub.add_parser(
        "transition", help="Apply a legal task status transition"
    )
    control_transition.add_argument("--state-root", default=".cosmo")
    control_transition.add_argument("--task-id", required=True)
    control_transition.add_argument("--status", required=True)
    control_transition.add_argument("--reason")
    control_transition.add_argument("--actor", default="operator")
    control_transition.set_defaults(func=control_command)

    control_list_plans = control_sub.add_parser(
        "list-plan-versions", help="List durable planning plan versions"
    )
    control_list_plans.add_argument("--state-root", default=".cosmo")
    control_list_plans.set_defaults(func=control_command)

    control_get_plan = control_sub.add_parser(
        "get-plan-version", help="Get one plan version by id"
    )
    control_get_plan.add_argument("--state-root", default=".cosmo")
    control_get_plan.add_argument("--plan-version-id", required=True)
    control_get_plan.set_defaults(func=control_command)

    control_list_integ = control_sub.add_parser(
        "list-integrations", help="List parent integration runs"
    )
    control_list_integ.add_argument("--state-root", default=".cosmo")
    control_list_integ.set_defaults(func=control_command)

    control_get_integ = control_sub.add_parser(
        "get-integration", help="Get one integration run by id"
    )
    control_get_integ.add_argument("--state-root", default=".cosmo")
    control_get_integ.add_argument("--integration-id", required=True)
    control_get_integ.set_defaults(func=control_command)

    control_assemble = control_sub.add_parser(
        "assemble-integration",
        help="Assemble parent integration workspace (no product mutation)",
    )
    control_assemble.add_argument("--state-root", default=".cosmo")
    control_assemble.add_argument("--repo-root", required=True)
    control_assemble.add_argument("--parent-node-id", required=True)
    control_assemble.add_argument(
        "--selected-json",
        help="Optional JSON object mapping leaf node_id -> artifact_version_id",
    )
    control_assemble.add_argument("--expected-baseline-tree-sha256")
    control_assemble.set_defaults(func=control_command)

    control_review_integ = control_sub.add_parser(
        "review-integration", help="Record independent integration review"
    )
    control_review_integ.add_argument("--state-root", default=".cosmo")
    control_review_integ.add_argument("--integration-id", required=True)
    control_review_integ.add_argument("--verdict", required=True)
    control_review_integ.add_argument("--notes")
    control_review_integ.set_defaults(func=control_command)

    control_promote_integ = control_sub.add_parser(
        "promote-integration",
        help="Explicitly promote an approved integration into product state",
    )
    control_promote_integ.add_argument("--state-root", default=".cosmo")
    control_promote_integ.add_argument("--integration-id", required=True)
    control_promote_integ.add_argument("--repo-root")
    control_promote_integ.set_defaults(func=control_command)

    control_inspect = control_sub.add_parser(
        "inspect-repo",
        help="Bounded repository inspection under an approved product root",
    )
    control_inspect.add_argument("--state-root", default=".cosmo")
    control_inspect.add_argument("--repo-root", required=True)
    control_inspect.add_argument(
        "--mode",
        dest="inspect_mode",
        choices=("map", "list", "read", "search", "symbols"),
        default="map",
        help="Inspection mode (default: map)",
    )
    control_inspect.add_argument("--path", help="Repo-relative path for read/symbols")
    control_inspect.add_argument("--prefix", help="Optional prefix for list/search")
    control_inspect.add_argument("--query", help="Literal search query")
    control_inspect.add_argument("--max-entries", type=int)
    control_inspect.add_argument("--max-bytes", type=int)
    control_inspect.add_argument("--max-matches", type=int)
    control_inspect.set_defaults(func=control_command)

    return root


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
