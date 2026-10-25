# Gemma 4 iterative self-verification calibration

Status: sealed development-pack evidence; not a held-out capability claim

This is the one archived example kept in the public tree. It used Cosmo's
first (v0.1.0) critique/amend protocol, in which the critic returned a `PASS`
or `STALLED` verdict; the current harness instead stops when the model returns
the current artifact unchanged (see `docs/ITERATIVE_SELF_VERIFICATION.md`).
It is development evidence for the mechanism, not a benchmark claim.

Run: `20260724T153723Z`

## Result

| Arm | Correct | Calls | Inference seconds | Output tokens | Reasoning tokens |
|---|---:|---:|---:|---:|---:|
| instant one-shot | 6/12 | 12 | 11.45 | 226 | 0 |
| instant recursive | 8/12 | 33 | 75.53 | 2,760 | 0 |
| thinking one-shot | 12/12 | 12 | 137.40 | 6,625 | 6,451 |
| thinking recursive | 12/12 | 24 | 245.52 | 11,712 | 11,176 |

Instant recursion repaired two one-shot failures: `trace-aliasing-002` and
`trace-state-003`. It did not repair the other four failures. Its critic issued
`PASS` on four incorrect terminal candidates among eleven `PASS` decisions, and
one run exhibited recurrence. It therefore improved this instant baseline, but
did not provide a dependable verification oracle.

Native thinking solved every case in this small visible development pack.
Adding recursive critique/amend preserved 12/12 accuracy while adding 12 calls,
108.12 inference seconds, and 4,725 reasoning tokens. Three recursive-thinking
critiques violated the verdict contract; the already-correct terminal
candidates were retained and the protocol failures were recorded.

The useful current signal is narrower than either headline hypothesis:
same-model recursive critique can improve an instant answer, but native thinking
one-shot was more accurate than instant recursion on this pack (12/12 versus
8/12) with fewer calls (12 versus 33). It was not cheaper in time or tokens: it
used 137.40 versus 75.53 inference seconds and 6,625 versus 2,760 output
tokens.
Recursive critique added no accuracy when native thinking was already at the
pack ceiling. Held-out, harder, and matched-resource cohorts are required before
generalizing.

## Protocol

- Model: `google/gemma-4-26b-a4b-qat`, MLX 4-bit
- Loaded instance: `cosmo-gemma4-26b`, context 8,192, parallelism 1
- LM Studio: `0.4.19+2`
- Sampling: temperature 0, maximum 2,048 output tokens
- Recursive ceiling: four critique/amend cycles
- Semantic stopping: same-model `PASS` or `STALLED`
- Harness stopping: hard cycle ceiling or protocol failure only
- Hidden evaluation: terminal and post-hoc; never returned to the model
- Arm order: deterministic shuffle with seed `20260724`

The mandatory capability probe observed zero reasoning tokens with reasoning
off and 312 reasoning tokens with reasoning on.

## Provenance

- Cosmo source commit:
  `219fa939c0e626da86ee631bb898ed29dcf6eecb`
- Source worktree at run start: clean
- Benchmark SHA-256:
  `f70f00b45b9734fd3ace3336f6bbdc0a86defcf4173bf426292ef585b9ed1dd1`
- Model artifact SHA-256:
  `cb567cbcd9f874d2daf1cc3faa8978ece122799892d47a5fd053a2d447e513ad`
- Chat-template SHA-256:
  `f41108adadfd4a8aa67c49731c4e121fecb7f19cd63fc567e65eb4f09d8b1cbd`
- Sealed run SHA-256:
  `c4e993a389a48f59df1041aa7463d8a078523fe6be3690842f619b76d974dd7d`

The machine-readable comparison is
`evidence/gemma4-26b-iterative-self-verification-20260724.json`. The complete manifest,
events, 48 per-case result records, summary, and seal receipt are under
`evidence/runs/20260724T153723Z`.

For publication, absolute local paths inside `manifest.json` and
`summary.json` were rewritten to repo-relative form. The seal receipt keeps the
original content hash (`sha256`, over the pre-redaction bytes) and adds
`publication_redaction.published_sha256`, which `cosmo.provenance.run_content_hash`
reproduces over the files as published.
