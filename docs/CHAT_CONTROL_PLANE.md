# Chat control plane (primary local interface)

Cosmo's primary interactive surface is `cosmo chat`: an ephemeral local chat
gateway over a **strict tool catalog** that acts only on the durable control
plane, planning service, context compiler, atomic harness, and parent
integration service. Inference is **local LM Studio only**.

This document describes the **implemented product-facing slice**.

## Primary interface

```bash
# Smoke: list tool contracts (no model call)
.venv/bin/cosmo control tools

# Smoke: inspect ready queue / tasks under a state root
.venv/bin/cosmo control ready-queue --state-root .cosmo
.venv/bin/cosmo control list-tasks --state-root .cosmo

# Bounded repository inspection (approved product root only)
.venv/bin/cosmo control inspect-repo \
  --state-root .cosmo \
  --repo-root /path/to/product \
  --mode map

# Plan versions / integrations
.venv/bin/cosmo control list-plan-versions --state-root .cosmo
.venv/bin/cosmo control list-integrations --state-root .cosmo

# One-shot chat turn (requires loaded LM Studio model)
.venv/bin/cosmo chat \
  --model <loaded-instance-id> \
  --repo-root /path/to/approved/product \
  --message "List my tasks." \
  --state-root .cosmo \
  --journal

# Interactive terminal REPL (stdin is a TTY, no --message)
.venv/bin/cosmo chat --model <loaded-instance-id> \
  --repo-root /path/to/approved/product --state-root .cosmo
```

- **Transcript**: in-memory only on the chat session; not product state.
- **Durable state**: tasks, specs, plan versions, leases, attempts, ledger,
  integration runs, and append-only control events under `--state-root`.
- **Model calls**: optional journaling via `--journal` →
  `state-root/chat-inference-journal` (`JournaledInferenceClient`).
- Model-backed tools use the loaded chat model by default; a tool may supply an
  explicit local model override. A dispatcher without either fails closed.
- **Tool-loop ops**: journaled as append-only `chat_tool_call` events (and
  `chat_events` records) with operation id, tool name, success/error, and
  task/attempt ids when available — **not** prompt or transcript bodies.
- **Caller provenance**: the trusted chat entry point authenticates a model
  principal out-of-band. Operator-only fallbacks require an authenticated
  operator object that cannot be constructed from tool JSON. Operator-supplied
  records persist that provenance.
- **Repository authority**: `--repo-root` is resolved once, approved by the
  operator, frozen into a caller-bound capability with its filesystem identity,
  and injected by the dispatcher. Model schemas contain no `repo_root`
  argument. Missing grants and post-grant root replacement fail closed.

### Input modes

| Mode | When | Behavior |
|---|---|---|
| One-shot | `--message` supplied | Handle that message and exit |
| REPL | no `--message`, stdin is a TTY | Multi-turn over one `ChatSession`; `/exit` or EOF ends |
| Pipe | no `--message`, stdin is not a TTY | Read all stdin as one message |

## Architecture

```text
User conversation (ephemeral)
      |
      v
cosmo chat  --strict tools only-->  ToolDispatcher
      |                                    |
      |                                    +--> ControlPlane
      |                                    +--> PlanningService + ContextCompiler
      |                                    +--> AtomicHarness (leaf attempts)
      |                                    +--> IntegrationService (parent join)
      v
Local LM Studio (InferenceClient)   Durable AtomicStore (.cosmo)
```

There is **no** second lifecycle/promotion authority. Chat never promotes
product state except through `promote_attempt` / `promote_integration` gates.

## Split strict tool catalogs

Tools are the only action channel. There is **no shell**, no HTTP provider
bridge, no cloud client, and no agent CLI. Only the reduced **model** catalog is
embedded in the system prompt. `cosmo control tools` emits both the model and
operator catalogs for inspection. Runtime dispatch selects the catalog again
from authenticated caller provenance; hiding a field in the prompt is not the
security boundary.

### Request / triage / task / attempt

| Tool | Authority |
|---|---|
| `submit_request` / `ingest_request` | Model: immutable Request intake only; operator catalog retains the explicit leaf shortcut |
| `propose_triage` / `review_triage` / `apply_triage` | Model catalog invokes local model-backed proposal/review only; operator payload fields exist only in the operator catalog |
| `create_task` | Operator-only leaf shortcut via `ControlPlane` (not `AtomicHarness.submit`) |
| `scheduler_tick` / `ready_queue` | Deterministic priority queue with aging + concurrency |
| `reassess_priority` | Append new explainable priority assessment (no product write) |
| `dispatch_scheduler_batch` | `scheduler_tick` then lease+dispatch selected ready leaves |
| `refresh_dependency_readiness` | blocked↔ready from plan dependency satisfaction |
| `inspect_lifecycle` | Task lifecycle projection or control-plane snapshot (read-only) |
| `dispatch_ready` / `run_attempt` / `retry_attempt` | `ControlPlane.dispatch_ready_leaf` → `AtomicHarness.run_attempt` |
| `promote_attempt` | Capability-bound `AtomicHarness.promote` gates only |
| `recover_promotion` | Operator-only recovery surface |

Optional `read_only_paths` may be supplied on dispatch lifecycle tools.

### Repository / context inspection

Bounded, approved-repo-root only: `repository_map`, `list_tree`, `read_artifact`,
`search_text`, `inspect_python_symbols`. Their model contracts contain no root
parameter; the dispatcher injects the immutable capability root. Unknown
capabilities, capability theft, explicit root overrides, traversal, and paths
outside the approved root fail closed before access. Handlers return JSON
records only — never live `RepositoryInspector` objects.

### Planning lifecycle

| Tool | Role |
|---|---|
| `build_planning_pack` | Deterministic pack + context manifest (no model) |
| `propose_plan` | Local planner via dispatcher `InferenceClient` + model |
| `review_plan` | Model catalog: independent local model review; operator review payload exists only in the operator catalog (never applies) |
| `list_plan_versions` / `get_plan_version` | Read durable plan versions |
| `apply_approved_plan` | Materialize nodes only after review PASS |

Proposal, independent review, and apply remain separate. Planning tools never
silently apply a plan and never instantiate a cloud client.

### Integration lifecycle

| Tool | Role |
|---|---|
| `assemble_integration` | Isolated baseline snapshot + overlay node-approved leaves |
| `review_integration` | Model catalog: local model-backed review only; verdict/evidence fallback is operator-only |
| `promote_integration` | Explicit product write under promotion lock |
| `list_integrations` / `get_integration` | Immutable run, checks, receipts |

Product files remain unchanged until `promote_integration` succeeds. Nested
parent children require a prior approved `IntegrationRun`; the ancestor uses
that run's approved content-addressed bytes rather than silently flattening or
rerunning grandchildren.

Malformed arguments, unknown tools, and handler failures return structured
`ToolResult` errors (fail-closed).

## Lifecycle and safety boundary

```text
Request -> (triage propose/review/apply) -> Task + TaskSpec + plan node
        -> (multi-artifact) propose_plan -> review_plan -> apply_approved_plan
        -> scheduler_tick / ready_queue / exclusive lease
           (reassess_priority, refresh_dependency_readiness, inspect_lifecycle)
        -> run_attempt | dispatch_ready | retry_attempt | dispatch_scheduler_batch
        -> promote_attempt (leaf) OR
           assemble_integration -> review_integration -> promote_integration
```

- Product files change **only** on successful promotion after Validation PASS
  and Review PASS (leaf) or integration validation+review PASS (parent).
- Worker sandboxes are **logical** (path/tool policy). Portable OS process
  jailing is not claimed.
- Control-plane status transitions are legal-edge only and append-only.
- Cosmo never exposes hidden evaluator data to model invocations.

## Local LM Studio requirement

Cosmo is local-only. Configure the operator-owned loopback LM Studio server
(`--transport lmstudio`, default). External providers and agent CLIs are out of
bounds for this harness.

```bash
.venv/bin/cosmo doctor
.venv/bin/cosmo doctor --model <instance> --probe
```

## Operator CLI control surfaces

In addition to chat tools, `cosmo control` exposes durable surfaces without a
model:

- `list-plan-versions` / `get-plan-version`
- `list-integrations` / `get-integration`
- `assemble-integration` / `review-integration` / `promote-integration`
- `inspect-repo` (`map` | `list` | `read` | `search` | `symbols`)

Unsafe path escapes are refused.

## Request triage (implemented)

Broad outcomes are not forced into a leaf artifact:

```text
submit_request / ingest_request  (immutable Request only)
      -> propose_triage          (model-backed or operator proposal)
      -> review_triage           (independent; PASS | RETRY_TRIAGE |
                                 NEEDS_USER_CLARIFICATION | ESCALATE)
      -> apply_triage            (create/link/update Task + TaskSpec)
      -> (when multi-artifact) propose_plan -> review_plan -> apply_approved_plan
      -> scheduler_tick / ready_queue -> leaf dispatch
```

Relationship taxonomy: `NEW`, `DUPLICATE_OF`, `OVERLAPS_WITH`, `DEPENDS_ON`,
`SUPERSEDES`, `CONFLICTS_WITH`. Priority is stored as explainable components
(spec names plus legacy aliases) with scheduler aging.

Operator-supplied triage/plan/integration review remains an explicit fallback,
requires authenticated operator provenance, and is recorded with
`operator_supplied=true` plus `caller_provenance`; it is not reachable from the
model catalog or presented as autonomous model review. Model-backed triage uses
the dispatcher’s local `InferenceClient` only.

CLI surfaces (no model required for operator payloads):

```bash
.venv/bin/cosmo control ingest-request --request-text '...'
.venv/bin/cosmo control propose-triage --request-id REQ-... --proposal-json '...'
.venv/bin/cosmo control review-triage --proposal-id TRIAGE-... --verdict PASS
.venv/bin/cosmo control apply-triage --proposal-id TRIAGE-... --repo-root /path
.venv/bin/cosmo control scheduler-tick --max-concurrent 2
```

## Deferred relative to the full specification

- Unattended multi-turn agent supervisors (REPL is operator-driven)
- Non-code external side-effect executors and research artifact pipelines
- Full synthesis / multi-candidate populations at every leaf
- Automatic JIT `expand_ready_child` scheduling (the API exists, but chat does
  not invoke it implicitly)
- Model-invoked worker tool loops and the full worker output envelope (the
  current leaf protocol is one complete artifact response)

See [ATOMIC_ARTIFACT_HARNESS.md](ATOMIC_ARTIFACT_HARNESS.md) for the end-to-end
artifact/attempt/integration spine shared with CLI `atomic` commands.
