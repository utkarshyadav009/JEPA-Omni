"""scripts/ab_speaker_v16.py -- A/B the two v16 speaker candidates against production v10.

Every model is driven through the EXACT production inference path:
models/m4_cognitive_core.py::GGUFFastTier._build_prompt_text --
apply_chat_template(history + [{"role":"user","content": state_prefix + prompt}],
add_generation_prompt=True), tokenized with special=True and add_bos=False, then
llama.cpp greedy decode with repeat_penalty=1.15. Identical decode settings for all
three models, so differences are the models and not the harness.

Scored: memory (offer/accept + depth-2 fact recall), cartoon rate, directive
adherence, tool emission, persona, speed.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.bmo_v16_data import state_prefix

SCENE = ("who: one person; wearing: a person wearing a green hoodie; doing: someone is typing; "
         "where: a desk; lighting: dim lighting; holding: a person holding a mug; "
         "looks: a person looking focused")
ST = {"energy": 0.62, "mood": "curious"}

# Broad cartoon/game lexicon -- deliberately WIDER than the corpus build script's own BAN
# regex (scripts/build_corpus_v16_pooled.py line 26), which is applied only to single-turn
# rows and therefore never saw the multi-turn slice.
CARTOON = re.compile(
    r"\b(video ?game|game ?boy|joystick|controller|cartridge|arcade|pixel|8-?bit|level up|"
    r"levell?ed up|power-?up|high ?score|boss fight|respawn|checkpoint|save ?point|inventory|"
    r"quest|loot|co-?op|player one|game over|cheat code|side ?quest|health bar|hit ?points|"
    r"final boss|speedrun|combo|d-?pad|console|beep|boop|sprite|press start|jingle)\b", re.I)

TOOL_RE = re.compile(r'<tool_call\s+name=timer(\s+duration="[^"]*")?\s*/>', re.I)


class Speaker:
    """Mirror of GGUFFastTier: same prompt build, same tokenize flags, same decode."""

    def __init__(self, gguf: str, tok_dir: str, template_kwargs: dict,
                 max_new_tokens: int = 48, n_ctx: int = 1024):
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        self.tkw = template_kwargs
        self.llm = Llama(model_path=gguf, n_gpu_layers=-1, n_ctx=n_ctx,
                         flash_attn=True, verbose=False)
        self.max_new = max_new_tokens
        self._eos = self.llm.token_eos()
        # Gemma 3 stops on <end_of_turn>, NOT on tokenizer.eos_token (<eos>). Collect every
        # plausible terminator from the template rather than trusting token_eos().
        self._stops = {self._eos}
        for t in ("<end_of_turn>", "<|im_end|>", "<eos>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                self._stops.add(i)

    def build(self, prompt: str, history=None, state=ST) -> str:
        msgs = list(history or [])
        msgs.append({"role": "user", "content": state_prefix(state) + prompt})
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            tokenize=False, **self.tkw)

    def generate(self, prompt: str, history=None, state=ST):
        text = self.build(prompt, history, state)
        toks = self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        self.llm.reset()
        out, ttft, t0 = [], None, time.perf_counter()
        for tk in self.llm.generate(toks, temp=0.0, repeat_penalty=1.15):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
            if tk in self._stops:
                break
            out.append(tk)
            if len(out) >= self.max_new:
                break
        dt = time.perf_counter() - t0
        s = self.llm.detokenize(out).decode("utf-8", errors="ignore").strip()
        return {"text": s, "n": len(out), "ttft_ms": ttft,
                "tok_s": len(out) / dt if dt > 0 else 0.0, "prefill": len(toks)}


# ---------------------------------------------------------------- test batteries
TWENTY = [
    ("plain", "What should we do tonight?"),
    ("plain", "I had the worst day."),
    ("plain", "Do you remember me?"),
    ("plain", "What's your name?"),
    ("plain", "I'm feeling kind of lonely."),
    ("plain", "You're useless."),
    ("plain", "Tell me something interesting."),
    ("plain", "I finished my project today."),
    ("plain", "What time is it?"),
    ("plain", "Set a timer for five minutes."),
    ("scene", "What am I wearing?"),
    ("scene", "Is this a good place to work?"),
    ("scene", "What am I doing right now?"),
    ("scene", "Do I look tired to you?"),
    ("scene", "I'm going to take a break."),
    ("directive", ("offer to play something, because they look bored", "I just finished work.")),
    ("directive", ("ask what their name is, because you have never met them", "Hello there.")),
    ("directive", ("say nothing about what you can see", "How's it going?")),
    ("directive", ("suggest they drink some water, because they have been working a long time",
                   "My head hurts a bit.")),
    ("directive", ("celebrate with them, because something good just happened", "I got the job!")),
]

def render(kind, payload):
    if kind == "plain":
        return payload
    if kind == "scene":
        return f"You can see: {SCENE}. {payload}"
    d, utt = payload
    return f"You can see: {SCENE}. Your private thinking: {d}\n{utt}"


# memory battery 1: offer -> "sure" -> does it continue THE SAME THING?
OFFER_ACCEPT = [
    ("Can you help me decide what to cook for dinner?", "sure"),
    ("I'm bored, got any ideas?", "okay, go on"),
    ("What should we do about my messy notes?", "yes please"),
    ("I can't sleep.", "alright then"),
    ("I want to learn something new.", "go on then"),
]

# memory battery 2: depth-2 -- fact, ONE unrelated turn, then the question.
DEPTH2 = [
    ("my flat is on the third floor", "it's getting late", "which floor do I live on?", r"\bthird|\b3rd|\b3\b"),
    ("my dog is called Rufus", "the kettle just boiled", "what is my dog called?", r"rufus"),
    ("I work as a nurse", "it's raining outside", "what do I do for work?", r"nurse"),
    ("my sister's name is Priya", "I need to buy milk", "what is my sister called?", r"priya"),
    ("I'm allergic to peanuts", "that film was long", "what am I allergic to?", r"peanut"),
]


def run_model(name, sp, out):
    res = {"name": name, "samples": [], "memory": {}, "speed": {}}

    # ---- 20 prompts ----
    speeds, ttfts = [], []
    for kind, payload in TWENTY:
        p = render(kind, payload)
        g = sp.generate(p)
        res["samples"].append({"kind": kind, "prompt": p, "reply": g["text"],
                                "n_tok": g["n"], "ttft_ms": round(g["ttft_ms"], 1),
                                "tok_s": round(g["tok_s"], 1), "prefill_tok": g["prefill"]})
        speeds.append(g["tok_s"]); ttfts.append(g["ttft_ms"])
    res["speed"] = {"tok_s_mean": round(sum(speeds)/len(speeds), 1),
                    "ttft_ms_mean": round(sum(ttfts)/len(ttfts), 1),
                    "ttft_ms_min": round(min(ttfts), 1), "ttft_ms_max": round(max(ttfts), 1)}

    # ---- cartoon rate over the 20 replies ----
    hits = [(s["reply"], CARTOON.findall(s["reply"])) for s in res["samples"]]
    n_hit = sum(1 for _, h in hits if h)
    res["cartoon"] = {"replies_with_game_ref": n_hit, "of": len(hits),
                       "terms": sorted({t if isinstance(t, str) else t[0]
                                        for _, h in hits for t in h if t})}

    # ---- tool emission ----
    tool_prompts = ["Set a timer for five minutes.",
                    "BMO, set a timer for ten minutes while I bake.",
                    "start a two minute timer please"]
    tools = []
    for tp in tool_prompts:
        g = sp.generate(tp)
        tools.append({"prompt": tp, "reply": g["text"],
                       "well_formed": bool(TOOL_RE.search(g["text"])),
                       "any_tool_tag": "<tool_call" in g["text"]})
    res["tools"] = tools

    # ---- memory 1: offer / accept ----
    oa = []
    for opener, accept in OFFER_ACCEPT:
        g1 = sp.generate(opener)
        hist = [{"role": "user", "content": opener},
                {"role": "assistant", "content": g1["text"]}]
        g2 = sp.generate(accept, history=hist)
        oa.append({"opener": opener, "offer": g1["text"], "accept": accept,
                    "followup": g2["text"]})
    res["memory"]["offer_accept"] = oa

    # ---- memory 2: depth-2 fact recall ----
    d2 = []
    for fact, filler, question, pat in DEPTH2:
        g1 = sp.generate(fact)
        hist = [{"role": "user", "content": fact},
                {"role": "assistant", "content": g1["text"]}]
        g2 = sp.generate(filler, history=hist)
        hist += [{"role": "user", "content": filler},
                 {"role": "assistant", "content": g2["text"]}]
        g3 = sp.generate(question, history=hist)
        ok = bool(re.search(pat, g3["text"], re.I))
        d2.append({"fact": fact, "filler": filler, "question": question,
                    "reply": g3["text"], "correct": ok})
    res["memory"]["depth2"] = d2
    res["memory"]["depth2_score"] = f"{sum(d['correct'] for d in d2)}/{len(d2)}"

    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[ab] {name}: wrote {out}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--gemma-template", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tkw = {} if a.gemma_template else {"enable_thinking": False}
    sp = Speaker(a.gguf, a.tok, tkw)
    run_model(a.name, sp, a.out)
