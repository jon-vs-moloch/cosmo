# Cosmo atomic-artifact harness

Status: Product-facing harness — local-only artifact control plane with leaf
attempts, context compilation, planning, narrow workers, logical sandbox, and
parent integration with explicit promotion.

Cosmo has deliberately separate surfaces:

1. **`cosmo chat` / `cosmo control` / `cosmo atomic`** — the artifact-centric
   product path (this document).
2. **`cosmo run` / `cosmo evolve`** — experimental reasoning/search sidecars.
   They are not the control-plane or product-acceptance spine.

## End-to-end product flow

```text
immutable Request  (ingest_request / submit_request without leaf contract)
      -> TriageService: propose -> independent review -> apply
         (relationships, requirements, priority components)
      -> versioned Task + TaskSpec + plan node
      -> (multi-artifact) PlanningService: propose -> independent review -> apply
      -> ContextCompiler builds short, role-specific manifests
      -> scheduler_tick / ready_queue (aging + concurrency limits + leases)
      -> ready leaf: Worker (one writable artifact) under SandboxPolicy
      -> deterministic Validation
      -> independent Review
      -> explicit leaf Promotion  OR  parent IntegrationService:
              assemble (baseline snapshot + child overlay)
           -> composition review
           -> explicit promote under promotion lock
```

## Model output transport

Planning and triage accept exactly one JSON object, optionally enclosed by one
complete outer Markdown JSON code fence. Surrounding whitespace is allowed;
prose prefixes or suffixes, multiple JSON values, and partial fences fail
closed. A Worker may receive the same single outer-fence normalization for its
replacement artifact, but patch/diff encodings are rejected before any
candidate bytes are written to its workspace.

Leaf shortcut: `create_task` / `submit_request` with objective + artifact still
uses operator-supplied triage defaults for single-artifact work. Broad chat
requests should not skip triage into a forced leaf.

The state root (default `.cosmo/`) is **search state**, not product state. It
holds immutable request/spec/attempt/artifact/validation/review records, plan
versions, integration runs, derived projections, an append-only event log, and
content-addressed candidates. Workers never write the product repository.
Promotion is the only path that mutates accepted product files.

## Local-only runtime

All model invocations use the operator-owned **LM Studio** loopback transport
via `InferenceClient`. Cosmo does not configure cloud providers. Triage,
planning, and review tools use the dispatcher-supplied local client and model
argument. Deterministic tests use fake clients; the suite does not require a
live LM Studio server.

## Planning and context compiler

`PlanningService` builds a deterministic planning pack (`ContextCompiler` +
optional repository map), proposes an immutable plan version, requires an
independent plan review, and only then materializes parent/leaf plan nodes.
Chat tools mirror that sequence; apply never happens as a side effect of
propose.

`ContextCompiler` records exact input manifests for planning, worker, and
review roles so later review can see the same contract the prior agent saw.
Before each attempt, the harness rebuilds a compact convergence-ledger
projection and includes it in the fresh Worker manifest; retries therefore see
rejected strategies and retained evidence without replaying transcripts.

## Worker tools and logical sandbox

The bounded WorkerToolset exposes narrow structured reads (`list_tree`,
`read_artifact`, `search_text`, `inspect_python_symbols`) and a single
`write_candidate` slot. The current canonical `AtomicHarness.run_attempt`
worker protocol still asks the model for one complete artifact response; it
does not yet run a model-invoked tool loop. Toolset calls are therefore an
available pre-attempt/library surface, not a claim that every worker model can
call tools during generation.
`SandboxPolicy` is fail-closed and durable (policy hash evidence). Portable
**OS-level process isolation is not guaranteed** in this pure-Python app; the
policy explicitly records that limitation. Tool budgets (`max_bytes`,
`max_entries`, `max_matches`) are checked against policy **before** reads.

Attempt workspaces must live under the AtomicStore state root.

## Parent integration

`IntegrationService` composes **direct** node-approved leaf children only:

1. Snapshot/copy the full accepted product baseline (excluding `.git` and the
   AtomicStore root) into an isolated workspace under the state root.
2. Overlay selected content-addressed child artifacts in dependency order.
3. Treat undeclared writes as paths that **changed vs baseline** outside the
   declared apply set (unchanged baseline files are not writes).
4. Record checks, selected children, workspace path, and baseline hash.
5. Require independent review PASS before promotion.
6. Promote under the existing exclusive promotion lock: stage and validate all
   bytes first; roll back on later write/hash failure so product is never left
   partially applied.

Nested parent children are composed through their already-approved,
content-addressed `IntegrationRun`; they are never re-run or silently treated
as unreviewed grandchildren. The ancestor overlays the exact approved bytes
recorded by that nested run, or fails closed if the nested acceptance gate is
missing.

Integration promotion currently has in-process rollback under the promotion
lock. Leaf promotion has durable crash-recovery intents; parent integration
promotion records the same durable intent: the exact staged paths, content
hashes, baseline tree, and expected post-write tree hash are written as a
`prepared` `integration_promotion_intents` record **before** the first product
byte is replaced, and committed only after acceptance bookkeeping finishes. A
later promote call finalizes a prepared intent when the product tree already
equals its recorded post-write tree; any other observed tree falls through to
the gated path and fails closed as stale rather than guessing.

## Smoke test (leaf)

Use a disposable fixture repository:

```bash
.venv/bin/cosmo atomic submit \
  --state-root /tmp/cosmo-state \
  --repo-root /tmp/cosmo-fixture \
  --request-text 'Make the fixture define value 2.' \
  --objective 'Set value to 2.' \
  --artifact target.py \
  --validator python_syntax

.venv/bin/cosmo atomic run \
  --state-root /tmp/cosmo-state \
  --repo-root /tmp/cosmo-fixture \
  --task TASK-... \
  --model <loaded-instance> \
  --model-dir /path/to/model \
  --chat-template /path/to/chat_template.jinja \
  --strategy-summary 'replace the assignment while preserving the file contract' \
  --allow-dirty

.venv/bin/cosmo atomic promote \
  --state-root /tmp/cosmo-state \
  --repo-root /tmp/cosmo-fixture \
  --attempt ATTEMPT-...
```

`atomic run` does not alter product files. Retries require the four-field
retry contract. Promotion rechecks baseline staleness under the promotion lock.

## Parent integration (operator CLI)

```bash
.venv/bin/cosmo control assemble-integration \
  --state-root /tmp/cosmo-state \
  --repo-root /tmp/cosmo-fixture \
  --parent-node-id NODE-...

.venv/bin/cosmo control review-integration \
  --state-root /tmp/cosmo-state \
  --integration-id INTEG-... \
  --verdict PASS

.venv/bin/cosmo control promote-integration \
  --state-root /tmp/cosmo-state \
  --integration-id INTEG-... \
  --repo-root /tmp/cosmo-fixture
```

## Current validators

Dependency-free leaf validators: `non_empty`, `utf8`, `json`, `python_syntax`.
Request triage/linkage status is a separate `request_states` projection; the
original Request JSON is never rewritten. Unknown validators fail
deterministically. Parent plans may re-run these
concrete adapters against applied artifacts; unsupported composition validators
fail closed until an adapter is registered.

## Related docs

- [CHAT_CONTROL_PLANE.md](CHAT_CONTROL_PLANE.md) — primary chat/tools surface
- [MACOS_APP.md](MACOS_APP.md) — Finder-launchable local wrapper
