"""scripts/bmo_v16_data.py -- shared corpus renderer for the v16 pooled corpus.

Used by BOTH speaker candidates (Gemma-3-270M and LFM2.5-350M) so the two runs
differ ONLY in base model + chat template, never in how the data was shaped.

WHY THIS FILE EXISTS SEPARATELY FROM THE TRAINING SCRIPTS
---------------------------------------------------------
v16 is the first corpus with multi-turn rows (2,124 of 11,586). Every previous
training path -- including scripts/finetune_bmo_minicpm5_lora.py -- assumed a
(prompt, target) PAIR and would silently drop or flatten a `messages` row. The
whole point of this run is multi-turn memory, so the render path is written once,
asserted, and shared.

TWO ROW SHAPES
  single : {"text": <reply>, "prompt": <what BMO is given>, "state": {..}}
  multi  : {"messages": [{"role": "user"|"assistant", "content": ..}, ..], "state": {..}}

INFERENCE SHAPE THIS MUST MATCH (models/m4_cognitive_core.py GGUFFastTier
._build_prompt_text + scripts/bmo_showcase.py):

    msgs = list(history)                       # bare user utterances + BMO's own replies
    msgs.append({"role": "user",
                 "content": _state_prefix(state) + prompt})
    apply_chat_template(msgs, add_generation_prompt=True)

  NO system prompt. (scripts/finetune_bmo_minicpm5_lora.py DID prepend a
  BMO_PERSONA_PROMPT system turn -- inference never sends one. That was a real
  train/inference mismatch carried by v10; it is not reproduced here.)

  `prompt` at inference is one of exactly three shapes:
    directive : "You can see: <scene>. Your private thinking: <directive>\n<utterance>"
    scene     : "You can see: <scene>. <utterance>"
    plain     : "<utterance>"

  The state prefix lands on the LAST user message ONLY -- bmo_showcase.py line
  1293 appends the RAW transcript to history, not the constructed prompt, so
  older user turns in the context carry no state, no scene and no directive.
  Rendered here the same way.

STATE ON EMPTY-STATE ROWS
  1,100 of the multi-turn rows (all of multiturn_fact_recall) carry "state": {}.
  At inference the state dict is ALWAYS populated, so training those rows bare
  would teach the depth-2 recall behaviour -- the single most important
  behaviour in this run -- in a prompt shape that never occurs live. Empty-state
  rows are therefore rendered with a state prefix sampled (seeded) from the
  states that DO appear in the corpus. This is a rendering choice, not a corpus
  edit; the corpus file is untouched.

LABEL MASKING
  Every assistant turn is supervised; all template scaffolding and every user
  turn is masked to -100. Segments are tokenised separately and concatenated,
  so the prompt/target boundary is exact by construction rather than inferred
  from a substring search.
"""
from __future__ import annotations

import json
import random
from typing import Any

import torch
from torch.utils.data import Dataset


def state_prefix(state: dict) -> str:
    """Byte-identical to models/m4_cognitive_core.py::_state_prefix."""
    if not state:
        return ""
    parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in state.items()]
    return "[" + " ".join(parts) + "] "


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_conversations(rows: list[dict], seed: int = 0) -> tuple[list[list[dict]], dict]:
    """rows -> OpenAI-style message lists, ready for apply_chat_template.

    Returns (conversations, stats). stats carries the counts the run must report,
    including how many multi-turn rows actually made it through.
    """
    rng = random.Random(seed)

    # Pool of real states seen in the corpus, for the empty-state rows (see module docstring).
    state_pool = []
    for r in rows:
        st = r.get("state") or {}
        if st:
            state_pool.append(st)
    if not state_pool:
        state_pool = [{"energy": 0.5, "mood": "curious"}]

    convs: list[list[dict]] = []
    n_multi = n_single = n_skipped = 0
    n_state_filled = 0
    multi_turn_counts = []

    for r in rows:
        st = dict(r.get("state") or {})
        if not st:
            st = dict(rng.choice(state_pool))
            n_state_filled += 1

        if "messages" in r:
            msgs = [dict(m) for m in r["messages"]]
            # must end on an assistant turn to be trainable
            if not msgs or msgs[-1].get("role") != "assistant":
                n_skipped += 1
                continue
            # state prefix on the LAST user message only -- matches inference
            last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=None)
            if last_user is None:
                n_skipped += 1
                continue
            msgs[last_user]["content"] = state_prefix(st) + msgs[last_user]["content"]
            convs.append(msgs)
            n_multi += 1
            multi_turn_counts.append(sum(1 for m in msgs if m["role"] == "assistant"))
        else:
            target = r.get("text")
            if not target:
                n_skipped += 1
                continue
            user = r.get("prompt") or "Say something."
            convs.append([
                {"role": "user", "content": state_prefix(st) + user},
                {"role": "assistant", "content": target},
            ])
            n_single += 1

    stats = {
        "n_rows_in": len(rows),
        "n_single": n_single,
        "n_multi": n_multi,
        "n_skipped": n_skipped,
        "n_state_filled": n_state_filled,
        "n_assistant_turns_multi": sum(multi_turn_counts),
    }
    return convs, stats


class ConvDataset(Dataset):
    """Tokenises a conversation into (input_ids, labels) with every assistant
    turn supervised and everything else masked to -100.

    `turn_end` is the token string that terminates an assistant turn in this
    model's template -- "<|im_end|>" for LFM2.5, "<end_of_turn>" for Gemma 3.
    GEMMA 3 FOOTGUN: tokenizer.eos_token is <eos> (id 1) but the chat template
    ends every turn with <end_of_turn> (id 106). Training the model to emit
    <eos> would make it never stop at inference, because llama.cpp stops on the
    template's terminator. The terminator is derived from the template itself
    (see derive_turn_end) rather than from tokenizer.eos_token.
    """

    def __init__(self, convs: list[list[dict]], tokenizer, max_len: int,
                 turn_end: str, template_kwargs: dict | None = None):
        self.convs = convs
        self.tok = tokenizer
        self.max_len = max_len
        self.turn_end = turn_end
        self.tkw = template_kwargs or {}
        self.n_truncated = 0

    def __len__(self) -> int:
        return len(self.convs)

    def _segments(self, msgs: list[dict]) -> list[tuple[str, bool]]:
        """(text, supervised) segments whose concatenation is the fully
        rendered conversation."""
        segs: list[tuple[str, bool]] = []
        cursor = ""
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            pre = self.tok.apply_chat_template(
                msgs[:i], add_generation_prompt=True, tokenize=False, **self.tkw)
            assert pre.startswith(cursor), "chat template is not concatenative"
            segs.append((pre[len(cursor):], False))
            body = m["content"] + self.turn_end
            segs.append((body, True))
            cursor = pre + body
        return segs

    def encode(self, msgs: list[dict]) -> tuple[torch.Tensor, torch.Tensor, bool]:
        ids: list[int] = []
        labels: list[int] = []
        for text, supervised in self._segments(msgs):
            t = self.tok(text, add_special_tokens=False)["input_ids"]
            ids.extend(t)
            labels.extend(t if supervised else [-100] * len(t))
        truncated = len(ids) > self.max_len
        return (torch.tensor(ids[: self.max_len]),
                torch.tensor(labels[: self.max_len]),
                truncated)

    def __getitem__(self, idx: int) -> dict:
        ids, labels, trunc = self.encode(self.convs[idx])
        if trunc:
            self.n_truncated += 1
        return {"input_ids": ids, "labels": labels}


def derive_turn_end(tokenizer, template_kwargs: dict | None = None) -> str:
    """Read the assistant-turn terminator straight out of the model's own chat
    template instead of assuming it equals eos_token."""
    tkw = template_kwargs or {}
    a = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}], add_generation_prompt=True, tokenize=False, **tkw)
    b = tokenizer.apply_chat_template(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        tokenize=False, **tkw)
    assert b.startswith(a), "chat template is not concatenative"
    tail = b[len(a):]
    assert tail.startswith("a"), f"unexpected assistant rendering: {tail!r}"
    return tail[1:].rstrip("\n")


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, n), pad_id, dtype=torch.long)
    labels = torch.full((B, n), -100, dtype=torch.long)
    attn = torch.zeros((B, n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = b["input_ids"].shape[0]
        input_ids[i, :k] = b["input_ids"]
        labels[i, :k] = b["labels"]
        attn[i, :k] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
