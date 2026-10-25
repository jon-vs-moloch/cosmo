# Unattended evolution durability

Status: inference-replay safe; durable process owner still required

## Guarantees in protocol v3

Every capability probe, seed draft, revision cycle, and prompt-revisor call has
a stable logical operation ID. `inference-journal/` records an in-flight receipt
before the local LM Studio call and atomically records the validated `Generation`
before returning it to evolutionary search logic.

Evolution gives each local LM Studio operation one journal boundary. Ordinary
paired `cosmo run` experiments retain their configured short retry policy. A
timeout and successful retry therefore cannot be hidden beneath one evolution
receipt: retryable evolution work is replayed at the journal boundary after a
resume.

On resume:

- a completed operation is replayed from its receipt without another LM Studio
  call;
- completed off/on capability probes are replayed on resume and included in the
  lifecycle total;
- a generation-zero interruption restarts bootstrap while reusing completed
  seed and base-evaluation operations;
- an incomplete later generation rolls back to the last atomic checkpoint and
  reuses completed proposal/evaluation operations;
- rollback uses a persisted, idempotent move plan;
- unique journal receipts, not duplicated artifacts, are the canonical search
  cost ledger;
- a nonblocking run-directory lock rejects concurrent writers.

If the process dies while a local LM Studio request is in flight, Cosmo cannot
know whether the server completed it. The next resume marks the prior attempt
`outcome_unknown`, retries the logical operation, and exposes
`accounting_complete=false`. It does not manufacture zero usage.

Attempt receipts distinguish `not_dispatched`, `known_rejected`,
`outcome_unknown`, `response_invalid`, and `completed_with_usage`. An LM Studio
response that cannot pass model or usage validation makes accounting incomplete
even when the transport error is not retryable. When validated usage was
available before a later response-model failure, it remains attached to the
attempt receipt.

General JSON artifacts use same-directory temp writes, file flush, atomic
replace, and directory flush. A terminal checkpoint without a summary can
therefore reconstruct the summary on resume rather than treating a partial file
as complete.

## Remaining boundary

Cosmo does not yet install a LaunchAgent or another durable supervisor. Running
`cosmo evolve` in a normal terminal gives deterministic replay and exact
completed-operation accounting, but it will not restart itself when the owning
terminal or host process disappears.

Before describing a search as unattended, give it an OS-owned supervisor that:

1. starts from an immutable, secret-free run specification;
2. invokes `cosmo evolve` and then `--resume` for the same directory after an
   unexpected exit;
3. uses bounded restart backoff;
4. stops on terminal state or a nonretryable manifest/configuration error;
5. keeps stdout/stderr beside the search;
6. leaves credentials in the service environment rather than the run spec.

The intended macOS implementation is a per-search or generic repo-owned
LaunchAgent. Until that front door exists, do not substitute `nohup`, an agent
terminal, or an undocumented shell loop and call the result durable.
