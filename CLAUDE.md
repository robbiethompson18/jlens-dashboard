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
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a
  test suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.

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
  — per-layer J-lens readouts of hidden steps (multi-hop, brew, chain-9) on Gemma-4-31B / Qwen3.6-27B, matplotlib reports; has its own `CLAUDE.md`, `pyproject.toml` and `docs/available-lenses.md` (lens inventory). Not wired into this repo's code yet.
