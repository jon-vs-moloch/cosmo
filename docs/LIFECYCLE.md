# Lifecycle

Cosmo is published as source for research and reference. It is not a
managed install: no installer, updater, rollback, or lifecycle hook ships
with it. Install from source (`pip install -e '.[dev]'`) and update by
pulling a newer revision.

`.nest/repository.json` and the `docs/architecture/`, `docs/decisions/`, and
`docs/continuity/` stubs are the repository lifecycle profile (`library@1`)
used by Owl + Kestrel's source host. They hold no credentials or host
details and are intentionally public.
