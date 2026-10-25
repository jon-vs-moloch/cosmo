# Contributor and agent instructions

Cosmo is an experimental inference harness for local models. Its only live
inference transport is a loopback LM Studio server; `cosmo run --replay`
answers from a fixture for offline tests and smoke runs and is never evidence.
The `atomic` command is the primary artifact-centric harness; `run` and
`evolve` are reasoning/search sidecars and must not become the product-state
acceptance path.

- Keep experiments reproducible, paired, and observable.
- Never run benchmark candidates in a real project checkout.
- Never expose hidden evaluator data to a model invocation.
- Keep LM Studio calls stateless unless statefulness is the treatment being
  tested.
- Store every candidate checkpoint, but score it only after the arm terminates.
- Do not treat model self-critique as acceptance evidence.
- Add complexity only when an ablation can measure its contribution.
- Run `pytest` before proposing a change; the suite is offline.
- Repository layout follows the `library@1` profile recorded in
  `.nest/repository.json`: durable architecture notes go in
  `docs/architecture/`, dated decisions in `docs/decisions/`, and
  `docs/continuity/current.md` is a generated projection, not a work log.
