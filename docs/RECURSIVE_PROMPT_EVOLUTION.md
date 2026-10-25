# Recursive prompt evolution

Status: implemented development-search protocol

## Purpose

Cosmo evolves two distinct prompt populations:

1. **Revision prompts (`R`)** control the direct or witnessed instant revision
   loop selected for the search.
2. **Prompt-revisor prompts (`M`)** control how the same fixed model proposes
   mutations to both `R` and `M`.

Only `R` enters benchmark task inference. `M` operates outside task loops and
never changes the task context, model, draft prompt, revision envelope,
sampling settings, evaluator, or cycle budget.

## Search cycle

An epoch begins with the selected treatment's canonical revision prompt `R0`
and the seed prompt-revisor strategy `M0`.

1. Generate one fixed instant draft per optimization case.
2. Evaluate `R0` by replaying those exact drafts.
3. Ask selected `M` genomes to propose complete `R` and `M` replacements.
4. Evaluate every unique `R` child against the same fixed drafts.
5. Select `R` lexicographically by protocol validity, terminal correctness,
   verification behavior, and finally token use.
6. Credit each `M` by the signed selection credit of every `R` child it
   proposed.
7. Admit `M` children only as future mutation operators; they earn no credit
   until they produce evaluated `R` offspring in a later generation.
8. Repeat to a correctness target, preregistered generation, or plateau
   boundary.

The configured mutation-call count is exact. The scheduler cycles through both
ranked `R` parents and selected `M` parents, rotating its starting offsets each
generation. This prevents a nominal multi-parent population from silently
collapsing to mutations of parent zero when the call budget is smaller than the
Cartesian product. A bounded number of valid `M` children is admitted each
generation, and untried children receive exploration slots before older
strategies can occupy the whole population.

The immutable mutation kernel controls JSON serialization and requested
offspring counts. Evolvable `M` strategies cannot change the parser, reward
arithmetic, lineage rules, or budgets.

The prompt-revisor output ceiling defaults to the smaller of 8,192 tokens and
the fixed LM Studio model's advertised generation ceiling. An explicit
`--mutation-max-output-tokens` can override the safe default. Requests are
clamped against the exact local catalog context and a conservative estimate of
current input. Requested and effective ceilings are retained in inference
telemetry; actual token use remains charged by the reward.

## Reward

Cosmo reports the historical scalar revision fitness:

```text
terminal accuracy
- token_penalty_per_1k_per_case
  * (mean charged revision tokens per case / 1000)
```

Charged tokens are input tokens plus total output tokens, including native
reasoning tokens. Fixed seed-draft work is reported in total search cost but
omitted from per-`R` comparison because the same draft corpus is replayed for
every prompt.

Actual `R` selection is lexicographic:

```text
valid protocol
terminal successes
fewer false returns
fewer regressions and protocol errors
more corrections
fewer charged tokens
```

An `R` child receives accuracy delta first. If accuracy is tied, it receives
bounded aggregate credit for fewer false returns, more corrections, fewer
regressions, and fewer protocol errors. Token improvement is considered only
when those outcomes tie. A valid `M` slot receives zero until that genome acts
and earns its own credit; an invalid or duplicate slot receives the configured
negative invalid-mutation credit. Every requested child slot therefore counts.
A revisor cannot win by spraying many children and taking credit only for its
best.

Prompt-revisor selection maximizes mean requested-slot credit. Its configured
proposal-token penalty is a deterministic final tie-break, not a scalar large
enough for stochastic native-reasoning length to overwhelm child quality.
Inference seconds, return-current decisions, recurrence, budget exhaustion, and
protocol failures remain diagnostics.

Plateau detection uses this same lexicographic selection key rather than the
historical scalar fitness. A correctness or verification-behavior improvement
therefore resets the plateau even when it spends more tokens. A local model
request that is rejected as an invalid candidate prompt invalidates and fully
penalizes that genome; LM Studio outages and retry exhaustion still stop the
search rather than masquerading as cheap candidates.

## Information boundary

The prompt-revisor sees:

- the exact reward function and weights;
- the current `R` and `M` text;
- model-observable process telemetry such as token use, recurrence, stop
  behavior, mutation validity, and duplicate status;
- recent model-observable lineage process for that `M` strategy;
- the immutable mutation-output contract.

It does not see benchmark prompts, references, evaluator rules, case outcomes,
observed accuracy, fitness, or selection credit. The harness owns evaluation and
selection. This preserves the evaluator firewall while the operational mutation
kernel, real parent rotation, and behavior-aware harness selection address the
prior no-op collapse.

Once accuracy participates in selection, the optimization pack is training
data. Search-set gains are optimizer-fit evidence, not a capability claim. A
candidate intended for a claim must be frozen and evaluated once on a separate,
sealed held-out pack whose result is never used for further mutation.

## Storage and lineage

Each search lives under `searches/<search-id>/`:

- `manifest.json`: fixed runtime, model, benchmark, reward, budgets, and hashes;
- `events.jsonl`: append-only inference, mutation, evaluation, and selection
  events;
- `inference-journal/`: atomic request-start and completed-generation receipts
  keyed by stable logical operation IDs;
- `seeds/`: the fixed draft corpus;
- `prompts/revision/` and `prompts/revisor/`: content-addressed genomes and
  parent/proposer lineage;
- `evaluations/`: complete terminal task results per revision prompt;
- `proposals/`: raw prompt-revisor generations, parsed offspring, rejected
  slots, and signed child deltas;
- `notifications/`: at-most-once local notification attempt receipts;
- `rollback/` and `orphaned/`: idempotent rollback plans and artifacts beyond
  the last committed generation;
- `state.json`: atomic generation-boundary checkpoint and derived rankings;
- `summary.json`: champion, complete leaderboards, and total search work.

Use `cosmo seal-evolution` to copy a completed search into
`evidence/searches/` with a content-hash receipt.

An interrupted search can be resumed with the original arguments plus
`--resume searches/<search-id>`. Runtime, source, model, benchmark, reward, and
budget mismatches fail closed. Artifacts from an incomplete generation are
moved under `orphaned/` by an idempotent rollback plan before execution
continues from the last atomic generation checkpoint. If generation zero never
reached a checkpoint, bootstrap restarts from its persisted operation receipts.
Completed capability-probe, seed, revision, and prompt-revisor operations are
replayed without another LM Studio call. Each local operation has one journal
boundary, so no timeout/retry can be hidden inside a completed receipt. A
request that was in flight at death cannot be proven complete; its receipt is
marked outcome-unknown before a retry. A post-response model or usage
validation failure is separately marked response-invalid and also makes
accounting incomplete. Search work includes preflight probes and is derived
from unique journal receipts rather than double-charging orphaned and replayed
artifacts.

The CLI holds a nonblocking single-writer lock for each evolution directory.
It does not yet install a durable background process owner, so a foreground
terminal remains resumable rather than self-restarting. See
`docs/UNATTENDED_EVOLUTION_DURABILITY.md` before launching a long local search.

## Unattended targets and milestones

`--target-correct N` stops only after a complete generation boundary when one
fully evaluated revision genome reaches at least `N` correct cases.
`--no-generation-ceiling` removes the fixed generation cap, allowing the search
to continue until the target, an enabled plateau/collapse rule, a fatal
transport error, or operator interruption.

Milestones use the highest raw success count ever observed, not the historical
scalar fitness. The deterministic accuracy leader is selected by successes,
then scalar fitness, then fewer tokens, then genome ID. The generation-0 score
establishes the baseline without backfilling notifications. A jump from 6/12 to
8/12 durably records both the 7/12 and 8/12 crossings.

With `--notify-milestones`, Cosmo reserves a notification receipt before calling
macOS Notification Center. A process crash can therefore lose a popup in the
small reservation-to-delivery window, but it cannot duplicate one on resume.
Notification failures never stop the search; `state.json` and milestone events
remain canonical.

## Example

```bash
.venv/bin/cosmo evolve \
  --model cosmo-gemma4-26b \
  --model-dir /path/to/model \
  --chat-template /path/to/chat_template.jinja \
  --benchmark benchmarks/iterative-self-verification/v1/cases.jsonl \
  --generations 8 \
  --revision-population 4 \
  --revisor-population 3 \
  --mutation-calls-per-generation 6 \
  --revision-mutations-per-call 2
```

For a target-driven unattended search:

```bash
.venv/bin/cosmo evolve \
  --model cosmo-gemma4-26b \
  --model-dir /path/to/model \
  --chat-template /path/to/chat_template.jinja \
  --benchmark benchmarks/iterative-self-verification/v1/cases.jsonl \
  --no-generation-ceiling \
  --target-correct 12 \
  --notify-milestones \
  --revision-population 4 \
  --revisor-population 3 \
  --mutation-calls-per-generation 6
```

To evaluate a frozen champion without editing source:

```bash
.venv/bin/cosmo run \
  --model cosmo-gemma4-26b \
  --model-dir /path/to/model \
  --chat-template /path/to/chat_template.jinja \
  --benchmark /path/to/held-out.jsonl \
  --arms instant_recursive \
  --revision-prompt-file /path/to/champion.txt
```
