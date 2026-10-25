# Iterative self-verification experiment

Status: implemented development protocol

## Hypothesis

An instant answer followed by stateless direct-revision cycles may reach a
correct answer faster or more often than one native-reasoning response from the
same model. The risks are correlated self-approval, revision regressions, and
thrash.

The first implementation runs a 2x2 factorial rather than comparing only two
arms:

| Arm | Model reasoning | Process |
|---|---|---|
| `instant_once` | `off` | one draft |
| `thinking_once` | `on` | one draft |
| `instant_recursive` | `off` | draft, direct revision loop |
| `thinking_recursive` | `on` | draft, direct revision loop |

All calls are stateless (`store=false`). Every revision receives exactly:

1. the original task and output contract;
2. the current draft;
3. the fixed adversarial revision instruction.

It receives no critique record, reference answer, previous draft, growing
transcript, evaluator result, or prior reasoning.

`--revision-mode direct` requires the revision call to emit only that artifact.
`--revision-mode witnessed` gives the same reasoning-off revision call a
bounded serial workspace, then extracts the artifact from:

```json
{"verification":"compact independent check","candidate":"complete artifact"}
```

The witness is fresh each cycle and is not projected into the next call. All
witness tokens are charged. Candidate identity remains the sole model-owned
return-or-replace signal; there is no redundant harness-owned decision label.
When the provider supports JSON Schema structured output, Cosmo constrains only
this serialization envelope, not the verification content or candidate.

Witnessed protocol v3 makes the compact check operational rather than
decorative: derive a complete answer independently before comparing it with the
draft; track object identities and variable bindings separately for state
traces; enumerate complete feasible assignments for finite constraints; and do
not emit a replacement that the same verification has already shown invalid.

## Model-owned convergence

The same model owns the semantic stopping decision. A revision call treats the
current draft as untrusted and deliberately tries to find a concrete fault. Its
output has only two valid forms:

- the complete replacement draft, when it finds a fault;
- the current draft exactly unchanged, when it cannot find a concrete fault
  after trying.

The replacement is the revision; there is no separate critique or amendment
call. If a fault is found but the correction is uncertain, the model must still
return its best complete replacement rather than repeating the draft. Textual
identity with the current artifact is the sole semantic convergence event. This
is intentionally self-assessed:
native chain-of-thought is not given an external verifier while it reasons, so
the recursive treatment must not receive one either.

Artifact identity uses the same line-ending normalization as candidate hashing;
otherwise it is textual. A decorated, fenced, or case-changed artifact is a new
draft rather than a return-current decision.

The harness stops when the model reproduces the current candidate. Other exact
recurrences and two-cycles are recorded as thrash but do not stop the loop. The
sole additional harness-owned termination is the predeclared hard cycle budget;
hitting it is budget exhaustion, not semantic convergence. The default cycle
cap is four. No hidden score participates in an online stop.

## Terminal evaluation

After the model stops—or a safety cutoff fires—the hidden evaluator scores every
saved candidate checkpoint. It is applied equally to all arms, supports only
post-hoc comparison and diagnosis, and is never returned to the model.

Primary:

- terminal hidden root success.

Secondary:

- terminal partial score;
- best checkpoint and cycle-to-best;
- final-minus-best score;
- calls, tokens, reasoning tokens, inference seconds, and wall time;
- false return-current decisions and revision regressions when the evaluator
  can establish them;
- repeated candidates, two-cycles, and inference spent after the best
  checkpoint.

The best checkpoint is not substituted for the terminal output. Doing that
would give recursive arms an unshared external selector.

On one fixed local model and machine, inference seconds are the primary resource
proxy because they include repeated prompt prefill and generation. Input
tokens, output tokens, reasoning tokens, cached input, transport
attempts/retries, and calls remain separate diagnostics; call count alone is
not treated as compute cost. Missing LM Studio response-model identity or
input/output usage is a failed run, not zero-cost evidence.

## Fairness

Hold fixed:

- exact model identity and local LM Studio route;
- for local runs, model build, quantization, loaded instance, LM Studio version,
  and prompt template;
- task projection and output contract;
- sampling parameters and output caps;
- evaluator and benchmark hash;
- machine and endpoint.

Every claim-bearing run must first pass a live off/on capability probe and
record the exact source commit, prompt hashes, evaluator hash, benchmark hash,
model record, transport provenance, and provider-effective output ceilings.
Runs additionally record model-artifact, chat-template, LM Studio, and
loaded-instance hashes. Raw results must be sealed into durable evidence with a
content-hash receipt.

Cosmo generates one immutable initial draft per case and reasoning setting,
then copies that same generation record into every selected one-shot and
recursive arm in the group. Thus `instant_once` and `instant_recursive` share
checkpoint zero by construction, as do their reasoning-on counterparts; a
provider seed is provenance, not the pairing mechanism. Seed and arm order are
deterministically shuffled to reduce thermal and time-order effects. The first
report is natural-completion evidence. A later matched-resource view should
pair by inference seconds because not every runtime exposes equivalent
reasoning-token accounting.

## Development pack

The bundled pack contains exact-answer, program-trace, and constraint-following
tasks. It is intentionally small and visible to developers. It is useful for
proving harness mechanics and detecting gross effects, not for claiming broad
capability gains.

Before a headline result, create:

- a held-out pack whose expected answers are not used during prompt iteration;
- at least twelve tasks across three families and three replicates;
- one-file code-repair fixtures in guarded disposable workspaces;
- preregistered go/no-go thresholds.

Provisional go signal:

- positive paired terminal-success signal in at least two task families;
- no containment or evaluator leakage;
- terminal-regression rate no greater than 10%;
- median inference after the retrospectively best checkpoint no greater than
  one cycle.

## First direct-revision calibration

On the initial twelve-case development pack, artifact-only instant revision
scored 6/12 in 29.82 inference seconds. The matched instant one-shot arm scored
6/12 in 12.45 seconds, while native thinking one-shot scored 12/12 in 154.13
seconds. Direct revision made no correction: eleven cases returned their
current artifact, including five incorrect artifacts, and one case oscillated
until budget exhaustion.

This is a negative result for the current treatment, not a demonstrated model
capability boundary. Its sealed run record is archived privately and is not
included in the public repository.
