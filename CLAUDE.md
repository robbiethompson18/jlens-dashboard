# jlens-dashboard

Interactive J-lens dashboard: hover any token of a Qwen transcript to see logit/J-lens top-20
readouts across layers, and grep the readouts.

@~/.claude/personal-repo-rules.md

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

## Stack

- Python: `uv` for deps/venv, `ruff` for lint + format (line length 140), `ty` for types. Run
  `uv run ruff check . && uv run ruff format . && uv run ty check` before shipping.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.

## Shipping and deploying

After every major edit, ship (commit + push to `main`) and deploy the public site with
`scripts/deploy.sh` (Cloudflare Pages, <https://jlens-dashboard.pages.dev/>) without asking Robbie.
Other agents share this checkout; ship everything in the tree at once rather than stashing.

Always work on `main`; do not create or switch to feature branches. To avoid stepping on another
agent, work in a sibling clone (`~/repos/jlens-dashboard-2`, `-3`, made with `/extra-checkout`)
rather than branching, and `git pull --rebase` before pushing.

## Docs

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `docs/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `docs/local/*.md`.

See `~/.claude/personal-repo-rules.md` (imported above) for the full convention.

Current docs:
<!-- As docs are added under docs/, list them here, one per line: -->
<!-- - [Title — when to read](docs/foo.md) — short gloss -->

- [visualizer/ — the former `jlens-visualizer` repo, folded in as a git subtree (squashed, 2026-10-15)](visualizer/README.md)
  — per-layer J-lens readouts of hidden steps (multi-hop, brew, chain-9) on Gemma-4-31B /
  Qwen3.6-27B, matplotlib reports; has its own `CLAUDE.md`, `pyproject.toml` and
  `docs/available-lenses.md` (lens inventory). Not wired into this repo's code yet.
- [Weird-behavior hunt findings — read before picking transcripts to visualize, or when asked what is odd about Qwen3.6-27B](docs/weird-behaviors-findings.md)
  — ranked summary of 13 subagents' probes (6k calls, 2026-10-15): memorized anti-distillation
  prompt, operator-directed lying, tools+json_schema fabrication, glitch tokens, provider artifacts.
  Harness, per-category reports, paper notes and the gzipped call log live in `weird-behaviors/`.
- [WorkspaceBench tasks and the published readouts — read before adding a bench task to the viewer](docs/wsbench-tasks.md)
  — survey of `camilablank/olens-results` (only `chain_intermediates` has judged readouts; 8 arms,
  120 items + 23 seed-first twins) and one line per bench family; what `scripts/wsbench.py` builds
  and how to add a task.

## UI state

By default every UI control's state (dataset, transcript, lens, top-N, layers, search terms, pin,
scope, checkboxes) is mirrored into the URL hash so any view is shareable by copying the link; add
new controls to `syncHash`/`openTranscript` in `index.html` rather than keeping their state only in
JS.
