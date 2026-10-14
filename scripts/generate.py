"""Sample Qwen3.6-27B's replies to data/prompts.json through OpenRouter; the lens readout later teacher-forces them on a GPU.

    uv run python scripts/generate.py            # writes data/transcripts.json

Sampling through a provider is only for speed: the readout re-runs the exact tokens through the bf16 weights, so a
provider's quantization changes which text we get, not what the lens reads on it.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

MODEL = "qwen/qwen3.6-27b"
MAX_TOKENS = 3000  # reasoning + reply; replies that hit it are dropped, so transcripts stay a readable length

prompts = json.loads(Path("data/prompts.json").read_text())


async def sample(client: httpx.AsyncClient, item: dict) -> dict | None:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "reasoning": {"enabled": item["think"]},
    }
    response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=600)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    if choice["finish_reason"] != "stop":
        print(f"drop {item['id']}: finish_reason={choice['finish_reason']}")
        return None
    message = choice["message"]
    reasoning = (message.get("reasoning") or "").strip()
    if item["think"] and not reasoning:
        print(f"warn {item['id']}: asked to think but got no reasoning back")
    print(f"ok   {item['id']}: {len(reasoning)} chars reasoning, {len(message['content'])} chars reply")
    return {**item, "reasoning": reasoning, "reply": message["content"].strip()}


async def main() -> None:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(*(sample(client, item) for item in prompts))
    transcripts = [r for r in results if r]
    Path("data/transcripts.json").write_text(json.dumps(transcripts, indent=1, ensure_ascii=False))
    print(f"wrote {len(transcripts)} transcripts")


asyncio.run(main())
