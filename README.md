# Cosmo

Cosmo is an experimental harness for asking whether explicit process (draft,
stateless self-revision, independent review, gated promotion) makes moderately
capable local models do better work. It targets a fixed model served by a
local LM Studio instance, so inference time, token usage, and server behavior
stay inspectable and reproducible. It is a research tool: everything it has
produced is development evidence, not benchmark claims.

Python 3.11+, standard library only. Licensed under Apache-2.0.

## Quickstart (offline)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest            # the suite is fully offline
.venv/bin/cosmo --help
```

Replay smoke: run the full paired experiment on one case with no model
server. `--replay` answers every inference from a fixture file, so the run is
deterministic and is recorded as `transport: "replay"`, never as evidence.

```bash
.venv/bin/cosmo run \
  --replay examples/replay/trace-accumulator.json \
  --model replay-model \
  --benchmark benchmarks/iterative-self-verification/v1/cases.jsonl \
  --case trace-accumulator-001 \
  --run-root /tmp/cosmo-runs
```

In that fixture the instant draft is wrong, one stateless revision repairs
it, and the thinking draft is already correct. See `src/cosmo/replay.py` for
the fixture schema.

## Running against LM Studio

Real runs need LM Studio 0.4+ with its local server running and one model
loaded that advertises reasoning `off` and `on`. Cosmo never starts or loads
models itself, and `--base-url` must be loopback (`localhost`, `127.0.0.1`, or
`::1`). Set `LM_API_TOKEN` if your server requires authentication.

```bash
.venv/bin/cosmo doctor                                # list loaded reasoning models
.venv/bin/cosmo doctor --model <instance-id> --probe  # expect probe_distinguishes_reasoning: true

.venv/bin/cosmo run \
  --model <instance-id> \
  --model-dir /path/to/the/model-directory \
  --chat-template /path/to/chat_template.jinja \
  --benchmark benchmarks/iterative-self-verification/v1/cases.jsonl \
  --limit 2
```

Results go to `runs/<run-id>/` (`manifest.json`, `seed_drafts.json`,
`events.jsonl`, `results/*.json`, `summary.json`). `cosmo compare` merges
runs with identical fixed configuration, and `cosmo seal` copies a run into
`evidence/runs/` with a content-hash receipt. A dirty source tree is refused
unless `--allow-dirty` marks the run as non-claim.

## What is in here

- **Reasoning experiment** (`cosmo run`): a paired 2x2 of instant versus
  native reasoning, one shot versus a stateless revision loop. Each revision
  call sees only the task, output contract, current draft, and a fixed critic
  instruction, and returns either a replacement or the current draft
  unchanged; returning it unchanged is the only semantic stop. Hidden
  evaluation runs only after an arm terminates. See
  [docs/ITERATIVE_SELF_VERIFICATION.md](docs/ITERATIVE_SELF_VERIFICATION.md).
- **Prompt evolution** (`cosmo evolve`): evolves the revision instruction and
  the prompt that mutates it, with journaled, resumable inference. See
  [docs/RECURSIVE_PROMPT_EVOLUTION.md](docs/RECURSIVE_PROMPT_EVOLUTION.md)
  and [docs/UNATTENDED_EVOLUTION_DURABILITY.md](docs/UNATTENDED_EVOLUTION_DURABILITY.md).
- **Atomic artifact harness** (`cosmo atomic`, `cosmo chat`, `cosmo control`):
  triage, planning, one-artifact workers, deterministic validation,
  independent review, and explicit promotion over a strict tool catalog. See
  [docs/ATOMIC_ARTIFACT_HARNESS.md](docs/ATOMIC_ARTIFACT_HARNESS.md) and
  [docs/CHAT_CONTROL_PLANE.md](docs/CHAT_CONTROL_PLANE.md).
- **macOS app wrapper (optional)**: `scripts/package_macos_app.sh` builds an
  unsigned, Finder-launchable wrapper for local testing only. It is not
  signed, notarized, or auto-updated; the CLI is the supported way to use
  Cosmo. See [docs/MACOS_APP.md](docs/MACOS_APP.md).

## Limits

- The bundled benchmark is a twelve-case visible development pack. It is a
  calibration instrument, and the prompt-evolution search trains on it, so
  no result here is a held-out or general capability claim.
- One archived example is included:
  [evidence/2026-07-24-gemma4-iterative-self-verification.md](evidence/2026-07-24-gemma4-iterative-self-verification.md),
  a sealed run of an earlier protocol version on one local model. Treat it
  as mechanism evidence, not a benchmark result.
- Recursive arms spend more inference by design. Reports keep calls, tokens,
  reasoning tokens, retries, and inference time so accuracy is never read
  without its cost.
- Native reasoning text is never stored or fed forward (only its token count
  and hash). LM Studio may keep its own server logs, so benchmark inputs must
  not contain secrets.
- Long `evolve` searches can resume, but Cosmo does not supervise or restart
  itself. Run them under an OS service manager if they must survive
  unattended.

## Distribution and lifecycle

This repository is a source and research publication, not a managed install:
there is no installer, updater, or lifecycle hook. Install it from source as
shown above. See [docs/LIFECYCLE.md](docs/LIFECYCLE.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
