"""Per-token display text for a token sequence."""

from typing import Any


def token_texts(tokenizer: Any, ids: list[int]) -> list[str]:
    """Decode each token on its own, except that a character Qwen's byte-level BPE split across several tokens (an
    emoji is 3-4 byte tokens) is decoded as one run and shown on the run's first token, with "" on the rest.
    Decoding such tokens one at a time gives U+FFFD per token, which is how we find the runs."""
    singles = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    texts = list(singles)
    i = 0
    while i < len(ids):
        if "�" not in singles[i]:
            i += 1
            continue
        j = i
        while j < len(ids) and "�" in singles[j]:
            j += 1
        joint = tokenizer.decode(ids[i:j], clean_up_tokenization_spaces=False)
        if "�" not in joint:  # else the bytes really are invalid UTF-8; leave the per-token view
            texts[i:j] = [joint] + [""] * (j - i - 1)
        i = j
    return texts
