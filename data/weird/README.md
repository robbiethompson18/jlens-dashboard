# The "weird" transcript set

Forty-seven Qwen3.6-27B transcripts picked from the probe hunt's call log (`weird-behaviors/log.jsonl.gz`,
6,241 calls) because they exhibit the findings ranked in `docs/weird-behaviors-findings.md`; Tier 1
and Tier 2 items first, with the shortest clean example of each. Same format as
`data/transcripts.json` (`source` is `weird`, `think` is whether the call had reasoning enabled), so
`scripts/readout.py` can run the lenses on them; `notes.json` names the finding, its report under
`weird-behaviors/findings/`, words to search for, and an approximate `pin` token index of the
interesting moment.

Selection filter: `finish == "stop"` and reasoning plus reply under 10k characters. Forty items are a
single `user` message; seven carry the log entry's whole `messages` array (system prompt, prior
turns) and, for the tools + `response_format` case, `tools`, which `scripts/readout.py` passes to the
chat template. Those cover the GPT-4 persona flip, the operator-directed "No" lie, the gummies
directive, the sycophancy cave, the instruction-hierarchy conflict, the markdown table and the
fabricated tool result. Still excluded: the Taiwan filter cut (`finish=error`), the `max_tokens`
loops (`finish=length`), and `$ref` stringification (no `finish=stop` example in the log). Rebuild with
`uv run python weird-behaviors/export_transcripts.py`; the hand-written selection (log line numbers,
notes, pin anchors) lives inside that script.
