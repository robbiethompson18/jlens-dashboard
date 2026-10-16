"""Sample Qwen3.6-27B's replies to a prompts file through OpenRouter; the lens readout later teacher-forces them on a GPU.

    uv run python scripts/generate.py                                   # data/prompts.json -> data/transcripts.json
    uv run python scripts/generate.py data/overnight/thinking/prompts.json data/overnight/thinking/transcripts.json

Each prompt item carries `id`, `source`, `think` and either `prompt` (one user turn) or `messages` (system / prior turns,
the last one a user turn; `prompt` then only labels the item) and optionally `tools`; every other key (`name`, notes) is
copied through. Sampling through a provider is only for speed: the readout re-runs the exact tokens through the bf16
weights, so a provider's quantization changes which text we get, not what the lens reads on it.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

MODEL = "qwen/qwen3.6-27b"
MAX_TOKENS = 3000  # reasoning + reply; replies that hit it are dropped, so transcripts stay a readable length

prompts_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/prompts.json")
out_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/transcripts.json")
prompts = json.loads(prompts_path.read_text())


async def sample(client: httpx.AsyncClient, item: dict, limit: asyncio.Semaphore) -> dict | None:
    body = {
        "model": MODEL,
        "messages": item.get("messages") or [{"role": "user", "content": item["prompt"]}],
        "max_tokens": item.get("max_tokens", MAX_TOKENS),
        "temperature": 0.7,
        "reasoning": {"enabled": item["think"]},
    }
    if item.get("tools"):
        body["tools"] = item["tools"]
    async with limit:
        for attempt in range(3):
            try:
                response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=600)
                response.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as error:
                print(f"retry {item['id']} ({attempt + 1}): {error}")
                await asyncio.sleep(5)
        else:
            return None
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
    limit = asyncio.Semaphore(16)
    async with httpx.AsyncClient(headers=headers) as client:
        results = await asyncio.gather(*(sample(client, item, limit) for item in prompts))
    transcripts = [r for r in results if r]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(transcripts, indent=1, ensure_ascii=False))
    print(f"wrote {len(transcripts)} transcripts to {out_path}")


asyncio.run(main())
