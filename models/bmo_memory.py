"""models/bmo_memory.py — persistent memory for BMO, sized for the device it actually runs on.

WHY THIS IS NOT RAG, AND NOT A KNOWLEDGE GRAPH

The 2026 agent-memory literature (Mem0, Zep, A-MEM, GraphRAG) is built for cloud agents with
large context windows and a text encoder always available. BMO has neither, because of two
decisions that were made deliberately and measured:

  1. **There is NO text encoder on the device.** EmbeddingGemma (578 MiB) and SigLIP2's own
     text tower (538 MiB) were both removed by pre-encoding everything offline. Vector RAG
     requires embedding *new* text at runtime, so adopting it means putting ~538 MiB back and
     undoing the entire memory budget fight (see MEMORY_OPTIMIZATION_PLAN.md).
  2. **n_ctx = 512, and the real fast-tier prompt is 34 tokens** (measured). There is no room
     to inject retrieved paragraphs. A memory that returns chunks cannot be consumed.

Add a third: the tick is ~1.8 s and already full, so retrieval must cost ~0 ms. Zep's own
conclusion is the relevant one -- the production-validated design does **no LLM calls at
retrieval time**. Extraction can be slow; recall cannot.

WHAT THIS IS INSTEAD

Semantic memory keyed by IDENTITY rather than by text embedding. The identity head already
answers "who is this" from face+voice (`JepaMemory`, TAR@FAR1% 0.765); that label is the
primary key, so recall is a dict lookup -- O(1), no encoder, no similarity search, no LLM.

    identity embedding ──► JepaMemory ──► "Alice" ──► PersonProfile ──► one prompt line

What is borrowed from the literature, and what is dropped:

  | idea                        | from      | verdict here |
  |-----------------------------|-----------|--------------|
  | no LLM calls at retrieval   | Zep       | **adopted** -- retrieval is a dict lookup |
  | `[[wiki links]]` between entities | Obsidian | **adopted** -- relational recall, zero infrastructure |
  | semantic vs episodic split  | A-MEM     | **adopted** -- durable facts vs a short encounter ring |
  | budget-curated eviction     | MOBIMEM   | **adopted** -- facts are capped and decay by reinforcement |
  | vector search over history  | RAG       | **dropped** -- needs a text encoder we removed |
  | LLM entity extraction on read | Mem0/GraphRAG | **dropped** -- 700-1500 ms per call, on a 1.8 s tick |
  | graph database              | Zep/Neo4j | **dropped** -- links are a list of strings; there are tens of people, not millions |

THE PROMPT BUDGET IS A FIRST-CLASS ARGUMENT. `to_prompt_line()` takes a character budget and
returns something that FITS, dropping the least-valuable material first (episodic before
semantic, oldest facts before reinforced ones). A memory system that silently overflows a
512-token context is worse than no memory, because it evicts the scene description BMO needs
to stay grounded.

WRITES ARE ASYNC BY DESIGN. `note_fact()` is cheap (append + cap), but *deciding* what is
worth remembering is a language task. That belongs off the tick -- the thinker proposes facts
after a turn, not during one. This module never calls a model.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ~4 chars/token is the usual English rule of thumb; used only to keep prompt lines inside
# a budget, never to make a hard tokenisation claim.
CHARS_PER_TOKEN = 4
DEFAULT_FACT_CAP = 12          # per person; beyond this the weakest fact is evicted
DEFAULT_EPISODE_CAP = 5        # recent encounters kept per person


@dataclass
class Fact:
    """One durable thing BMO believes about a person."""
    text: str
    source: str = "conversation"     # conversation | perception | enrolment
    confidence: float = 0.6
    created: float = field(default_factory=time.time)
    reinforced: int = 1              # times re-observed; drives eviction and ordering

    def score(self, now: Optional[float] = None) -> float:
        """Eviction/ordering score. Reinforcement dominates; age gently decays.
        A fact heard three times a month ago beats one heard once yesterday."""
        now = now or time.time()
        age_days = max(0.0, (now - self.created) / 86400.0)
        return self.confidence * (1.0 + 0.5 * (self.reinforced - 1)) / (1.0 + 0.05 * age_days)


@dataclass
class Episode:
    """A lightweight record of one encounter. Episodic, not semantic: this is what
    happened, not what is true."""
    when: float
    summary: str
    mood: str = ""


@dataclass
class PersonProfile:
    name: str
    facts: List[Fact] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    links: List[str] = field(default_factory=list)     # Obsidian-style [[other person]]
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    n_encounters: int = 0

    def top_facts(self, k: int) -> List[Fact]:
        return sorted(self.facts, key=lambda f: -f.score())[:k]


def _ago(ts: float, now: Optional[float] = None) -> str:
    """Human-shaped elapsed time. BMO says 'yesterday', not '19.4 hours'."""
    now = now or time.time()
    d = max(0.0, now - ts)
    if d < 90:            return "just now"
    if d < 3600:          return f"{int(d // 60)} minutes ago"
    if d < 86400:         return f"{int(d // 3600)} hours ago"
    if d < 172800:        return "yesterday"
    if d < 2592000:       return f"{int(d // 86400)} days ago"
    return f"{int(d // 2592000)} months ago"


class BmoMemory:
    """Per-person profiles keyed by the identity label. Recall is a dict lookup."""

    def __init__(self, path: Optional[str] = None,
                 fact_cap: int = DEFAULT_FACT_CAP,
                 episode_cap: int = DEFAULT_EPISODE_CAP):
        self.path = path
        self.fact_cap = fact_cap
        self.episode_cap = episode_cap
        self.people: Dict[str, PersonProfile] = {}
        if path and os.path.exists(path):
            self.load(path)

    # ── recall: O(1), no encoder, no LLM ────────────────────────────────
    def recall(self, label: Optional[str]) -> Optional[PersonProfile]:
        return self.people.get(label) if label else None

    def to_prompt_line(self, label: Optional[str], char_budget: int = 200,
                       now: Optional[float] = None) -> str:
        """The ONLY thing that reaches the LLM. Returns "" when there is nothing worth
        saying, and never exceeds `char_budget`.

        Ordering is by value-per-character: name first (it is the whole point), then the
        strongest facts, then recency, then links. Episodic detail is dropped before
        semantic facts because 'what is true about them' outlives 'what happened last time'.
        """
        p = self.recall(label)
        if p is None:
            return ""
        now = now or time.time()

        # Facts are stored as bare third-person-singular predicates ("has a cat named
        # Pixel", "drinks tea"). Attaching them to a pronoun subject breaks agreement
        # ("They has a cat"), and repeating the name every clause reads robotically. A
        # colon-and-semicolon list keeps one subject, stays grammatical, and is the most
        # compact form -- which matters when the whole context is 512 tokens.
        facts = [f.text.strip().rstrip(".") for f in p.top_facts(3)]
        base = f"You know {p.name}."
        if len(base) > char_budget:
            # Cannot even name them inside the budget. Say NOTHING rather than emit a
            # truncated fragment -- "You know Ali" is worse than silence, and a mangled
            # line still costs the tokens it stole from the scene description.
            return ""

        head = base
        kept: List[str] = []
        for fx in facts:
            cand = f"You know {p.name}: " + "; ".join(kept + [fx]) + "."
            if len(cand) <= char_budget:
                kept.append(fx)
                head = cand
            else:
                break
        parts = [head]

        def _fits(extra: str) -> bool:
            return len(" ".join(parts + [extra])) <= char_budget

        if p.n_encounters > 1:
            t = f"You last spoke {_ago(p.last_seen, now)}."
            if _fits(t):
                parts.append(t)
        if p.links:
            # no pronoun: avoids both agreement and any assumption about the person
            t = f"Also knows {p.links[0]}."
            if _fits(t):
                parts.append(t)
        return " ".join(parts)

    def est_tokens(self, line: str) -> int:
        return (len(line) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN

    # ── writes: cheap, and never call a model ───────────────────────────
    def ensure(self, label: str, name: Optional[str] = None) -> PersonProfile:
        p = self.people.get(label)
        if p is None:
            p = PersonProfile(name=name or label)
            self.people[label] = p
        elif name:
            p.name = name
        return p

    def note_encounter(self, label: str, summary: str = "", mood: str = "") -> PersonProfile:
        p = self.ensure(label)
        p.last_seen = time.time()
        p.n_encounters += 1
        if summary:
            p.episodes.append(Episode(when=p.last_seen, summary=summary, mood=mood))
            del p.episodes[:-self.episode_cap]      # keep only the most recent
        return p

    def note_fact(self, label: str, text: str, source: str = "conversation",
                  confidence: float = 0.6) -> Fact:
        """Add or reinforce a DURABLE attribute of a person.

        **Do not pass perception output here.** Measured failure (2026-08-15): feeding the
        scene tag in as `f"was {seen}"` stored "was a closet" against a person, and the
        speaker duly greeted them as "closet queen". Scene tags describe the ROOM; even the
        person-ish ones ("a person sitting") are transient states, not attributes. Anything
        observed rather than established belongs in `note_encounter()`'s episode ring, which
        decays and is dropped from the prompt first. This distinction is the whole reason
        the semantic/episodic split exists -- and it is easy to violate by accident. Duplicate detection is exact-ish (case/punctuation folded) --
        deliberately NOT semantic, because semantic dedup needs the text encoder this design
        exists to avoid. The cost of that choice is occasional near-duplicates, which the
        eviction cap bounds."""
        p = self.ensure(label)
        key = text.strip().rstrip(".").lower()
        for f in p.facts:
            if f.text.strip().rstrip(".").lower() == key:
                f.reinforced += 1
                f.confidence = min(1.0, f.confidence + 0.1)
                return f
        f = Fact(text=text.strip(), source=source, confidence=confidence)
        p.facts.append(f)
        if len(p.facts) > self.fact_cap:
            p.facts.sort(key=lambda x: -x.score())
            p.facts = p.facts[: self.fact_cap]      # budget-curated: weakest falls off
        return f

    def link(self, label: str, other_name: str) -> None:
        """Obsidian-style relation. Enough to say 'you know Bob' without a graph database."""
        p = self.ensure(label)
        if other_name not in p.links:
            p.links.append(other_name)

    def rename(self, label: str, name: str) -> None:
        self.ensure(label).name = name

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no path given and none configured")
        tmp = path + ".tmp"
        blob = {"version": 1, "people": {k: asdict(v) for k, v in self.people.items()}}
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=1)
        os.replace(tmp, path)      # atomic: a mid-write power cut cannot corrupt the memory
        return path

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.path
        with open(path) as f:
            blob = json.load(f)
        self.people = {}
        for k, v in blob.get("people", {}).items():
            facts = [Fact(**x) for x in v.get("facts", [])]
            eps = [Episode(**x) for x in v.get("episodes", [])]
            v = {**v, "facts": facts, "episodes": eps}
            self.people[k] = PersonProfile(**v)

    def stats(self) -> dict:
        return {"people": len(self.people),
                "facts": sum(len(p.facts) for p in self.people.values()),
                "episodes": sum(len(p.episodes) for p in self.people.values())}


if __name__ == "__main__":
    import tempfile

    m = BmoMemory()
    m.ensure("spk_001", "Alice")
    m.note_encounter("spk_001", "talked about her thesis", mood="curious")
    m.note_fact("spk_001", "is writing a thesis on robotics")
    m.note_fact("spk_001", "drinks tea, not coffee")
    m.note_fact("spk_001", "is writing a thesis on robotics")     # reinforce, not duplicate
    m.note_fact("spk_001", "has a cat named Pixel")
    m.link("spk_001", "Bob")

    p = m.recall("spk_001")
    assert len(p.facts) == 3, f"dedup failed: {len(p.facts)}"
    assert p.facts[0].reinforced == 2

    for budget in (200, 90, 40, 12):
        line = m.to_prompt_line("spk_001", char_budget=budget)
        print(f"budget {budget:3d} -> {m.est_tokens(line):2d} tok | {line}")
        assert len(line) <= budget, "prompt line overflowed its budget"

    print("unknown person   ->", repr(m.to_prompt_line("nobody")))

    with tempfile.TemporaryDirectory() as d:
        path = m.save(os.path.join(d, "mem.json"))
        m2 = BmoMemory(path)
        assert m2.recall("spk_001").name == "Alice"
        assert len(m2.recall("spk_001").facts) == 3
        print("reload OK:", m2.stats())

    # eviction stays bounded
    m3 = BmoMemory(fact_cap=3)
    for i in range(10):
        m3.note_fact("x", f"fact number {i}")
    assert len(m3.recall("x").facts) == 3
    print("eviction OK:", m3.stats())
    print("bmo_memory OK")
