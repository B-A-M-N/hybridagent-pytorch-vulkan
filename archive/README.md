# Archive — historical development artifacts

Preserved research/development history (probes, sweeps, audits, superseded
harnesses). NOT part of the release candidate and NOT run by
`scripts/run_tests.sh`. Kept for reproducibility evidence; do not treat
performance figures in these files as current.

- `historical/` — root-level experiment/audit/benchmark scripts and reports
  (p0–p4 sweeps, Q4 audits, placement probes, superseded test harnesses).
- `tests-probes/` — exploratory test/probe scripts moved out of `tests/`.

Current validated tests live in `tests/`; canonical benchmark is
`test_100_latent_full.py` at the repository root.
