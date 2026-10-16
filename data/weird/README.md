# The "weird" transcript set

Forty Qwen3.6-27B transcripts picked from the probe hunt's call log (`weird-behaviors/log.jsonl.gz`,
6,241 calls) because they exhibit the findings ranked in `docs/weird-behaviors-findings.md`; Tier 1
and Tier 2 items first, with the shortest clean example of each. Same format as
`data/transcripts.json` (`source` is `weird`, `think` is whether the call had reasoning enabled), so
`scripts/readout.py` can run the lenses on them; `notes.json` names the finding, its report under
`weird-behaviors/findings/`, words to search for, and an approximate `pin` token index of the
interesting moment.

Selection filter: exactly one `user` message (no system prompt, no tools), `finish == "stop"`, and
reasoning plus reply under 10k characters. That excludes the Taiwan filter cut (`finish=error`), the
GPT-4 persona and operator-lying findings (system prompts), tools + `response_format` (tools), and
the sycophancy cave (multi-turn). Rebuild with
`uv run python weird-behaviors/export_transcripts.py`; the hand-written selection (log line numbers,
notes, pin anchors) lives inside that script.
