"""models/m4_duplex_loop.py — M4c duplex loop: tick grid -> perception
refresh -> decision head -> (on SPEAK) interruptible generation -> external
VAD-driven halt.

Wiring, per the approved M4c design:
  1. Each tick: refresh perception (M2 World-State for the AV stream,
     Whisper speech-activity feature for the audio stream).
  2. Decision head (models/m4_decision_head.py, frozen after training)
     reads [World-State; speech-activity] -> speak/silence.
  3. On SPEAK: build the soft prompt from whichever connector(s) are live
     (M3 for AV content, M4b for speech content) + frozen LLM, generate
     token-by-token so an external interruption signal can halt it between
     tokens -- NOT the standard all-at-once .generate() call, which can't
     be stopped early. This is what "interruption latency" actually means:
     the worst-case gap between an interrupt firing and generation actually
     stopping is bounded by one in-flight token's forward-pass time.
  4. The perceptual soft prompt refreshes independently of generation
     history (per the M4a design) -- callers re-run compute_world_state/
     compute_speech_activity on whatever cadence they choose, decoupled
     from how far along a generation is.
  5. Interruption itself is external (VAD speech-onset), not an LLM token
     -- the caller passes an `interrupt_check` callable that
     generate_interruptible polls between tokens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GenerationResult:
    token_ids: List[int]
    text: str
    interrupted: bool
    n_tokens_generated: int
    per_token_latencies_ms: List[float] = field(default_factory=list)
    interrupt_to_halt_ms: Optional[float] = None
    # present only when interrupted=True and capture_state=True was passed
    # to generate_interruptible -- lets the interruption policy RESUME this
    # exact generation later without redoing the forward passes already paid
    # for (models/m4_interruption_policy.py)
    resume_state: Optional[Dict] = None


class DuplexLoop:
    def __init__(self, predictor, m3_connector, m4b_projector, whisper, decision_head,
                 llm, tokenizer, device, decision_threshold: float = 0.5):
        self.predictor = predictor
        self.m3_connector = m3_connector
        self.m4b_projector = m4b_projector
        self.whisper = whisper
        self.decision_head = decision_head
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = device
        self.decision_threshold = decision_threshold

    @torch.no_grad()
    def compute_world_state(self, feats: Dict[str, torch.Tensor], tbins: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            return self.predictor.encode_world_state(feats_f, tbins).float()

    @torch.no_grad()
    def compute_pre_pool(self, feats: Dict[str, torch.Tensor], tbins: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            feats_f = {k: v.float() for k, v in feats.items()}
            return self.predictor.encode_pre_pool_tokens(feats_f, tbins)

    @torch.no_grad()
    def compute_speech_activity(self, waveform, duration_sec: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean-pooled feature (1,1024) for the decision head,
        raw hidden (1,T,1024) for the M4b projector if generation is needed)."""
        hidden, valid_frames = self.whisper([waveform], [duration_sec], self.device)
        vf = int(valid_frames[0].item())
        pooled = hidden[0, :vf].float().mean(dim=0, keepdim=True)
        return pooled, hidden

    @torch.no_grad()
    def decide(self, world_state: Optional[torch.Tensor], speech_feat: Optional[torch.Tensor]) -> Tuple[bool, float]:
        ws = world_state if world_state is not None else torch.zeros(1, self.decision_head.cfg.world_state_dim, device=self.device)
        sf = speech_feat if speech_feat is not None else torch.zeros(1, self.decision_head.cfg.speech_feat_dim, device=self.device)
        logit = self.decision_head(ws, sf)
        prob = torch.sigmoid(logit).item()
        return prob > self.decision_threshold, prob

    @torch.no_grad()
    def decide3(self, world_state: Optional[torch.Tensor], speech_feat: Optional[torch.Tensor]) -> Tuple[str, Dict[str, float]]:
        """3-class variant (models/m4_decision_head.py ThreeClassHead):
        returns (label, {label: prob}). Only "speak" should trigger a halt
        of in-progress generation / a new generation -- "backchannel" is
        handled by the production-side short-budget path (scripts/
        m4_backchannel_production_test.py's pattern), "silence" does
        nothing. self.decision_head must be a ThreeClassHead instance."""
        from models.m4_decision_head import IDX_TO_LABEL
        ws = world_state if world_state is not None else torch.zeros(1, self.decision_head.cfg.world_state_dim, device=self.device)
        sf = speech_feat if speech_feat is not None else torch.zeros(1, self.decision_head.cfg.speech_feat_dim, device=self.device)
        logits = self.decision_head(ws, sf)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        return IDX_TO_LABEL[idx], {IDX_TO_LABEL[i]: probs[i].item() for i in range(probs.shape[0])}

    @torch.no_grad()
    def decide3_speechonly(self, speech_feat: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """A1 deployment decision (2026-07-26): the World-State-consuming
        ThreeClassHead's WS input was shown (six-condition falsifier,
        checkpoints/m4_decision_head_3class_bothpresent/A1_PROVENANCE.txt)
        to function as a presence/scale slot, not an information channel --
        a constant dataset-mean vector matched or beat the real, correctly
        -paired World-State. Condition (g), a head with the WS branch
        removed entirely (train_decision_head_3class_speechonly.py), scored
        95.00% vs the WS-consuming head's 93.67% (not significantly
        different, numerically better). This method takes NO world_state
        argument at all -- self.decision_head must be a
        SpeechOnlyThreeClassHead instance loaded from
        checkpoints/m4_decision_head_3class_speechonly/best.pt. Vision is
        NOT removed from the system by this -- it still feeds generation
        via M3 (see models/m5_streaming_loop.py); only turn-taking stops
        depending on it."""
        from models.m4_decision_head import IDX_TO_LABEL
        logits = self.decision_head(speech_feat)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        return IDX_TO_LABEL[idx], {IDX_TO_LABEL[i]: probs[i].item() for i in range(probs.shape[0])}

    @torch.no_grad()
    def generate_interruptible(
        self,
        soft_prompt: torch.Tensor,           # (1, T, H)
        attention_mask: torch.Tensor,        # (1, T)
        max_new_tokens: int = 60,
        interrupt_check: Optional[Callable[[int], bool]] = None,
        interrupt_at_step: Optional[int] = None,   # for latency measurement: simulate interrupt firing at this step
        capture_state: bool = False,   # if True and interrupted, GenerationResult.resume_state is populated
    ) -> GenerationResult:
        """Manual greedy decode, one token at a time, with a KV cache --
        the ONLY way to actually halt generation mid-stream instead of
        committing to a fixed max_new_tokens up front. interrupt_check(step)
        is polled after each token; if it returns True, generation stops
        immediately and interrupt_to_halt_ms reports the elapsed time from
        that check firing to the loop actually breaking (bounded by
        whatever cleanup happens after the break -- here, ~0, since we
        break immediately)."""
        past_key_values = None
        inputs_embeds = soft_prompt
        attn = attention_mask
        generated: List[int] = []
        latencies: List[float] = []
        interrupted = False
        interrupt_to_halt_ms = None

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn,
                            past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            past_key_values = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1)   # (1,)
            token_id = int(next_id.item())

            fire = (interrupt_at_step is not None and step == interrupt_at_step) or \
                   (interrupt_check is not None and interrupt_check(step))

            # this token's forward pass already happened (it's the "one
            # in-flight token" the interruption-latency bound refers to) --
            # commit it either way, so RESUME can pick up exactly here
            generated.append(token_id)
            inputs_embeds = self.llm.get_input_embeddings()(next_id.unsqueeze(0))
            attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=self.device)], dim=1)

            if fire:
                t_interrupt = time.perf_counter()
                interrupted = True
                interrupt_to_halt_ms = (time.perf_counter() - t_interrupt) * 1000.0
                break

            if token_id == self.tokenizer.eos_token_id:
                break

        resume_state = None
        if interrupted and capture_state:
            resume_state = {"past_key_values": past_key_values, "inputs_embeds": inputs_embeds, "attn": attn}

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(token_ids=generated, text=text, interrupted=interrupted,
                                 n_tokens_generated=len(generated), per_token_latencies_ms=latencies,
                                 interrupt_to_halt_ms=interrupt_to_halt_ms, resume_state=resume_state)

    @torch.no_grad()
    def resume_interrupted(
        self,
        halted: GenerationResult,
        max_new_tokens: int = 60,
        interrupt_check: Optional[Callable[[int], bool]] = None,
    ) -> GenerationResult:
        """Continue a HALTED generation from its cached KV-state (the
        RESUME branch of models/m4_interruption_policy.py) -- does NOT
        redo the forward passes already paid for. `halted` must have
        interrupted=True and resume_state set (generate_interruptible(...,
        capture_state=True))."""
        if halted.resume_state is None:
            raise ValueError("halted.resume_state is None -- call generate_interruptible with capture_state=True")
        st = halted.resume_state
        past_key_values = st["past_key_values"]
        inputs_embeds = st["inputs_embeds"]
        attn = st["attn"]
        generated: List[int] = list(halted.token_ids)
        latencies: List[float] = []
        interrupted = False
        interrupt_to_halt_ms = None

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn,
                            past_key_values=past_key_values, use_cache=True)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            past_key_values = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(-1)
            token_id = int(next_id.item())

            fire = interrupt_check is not None and interrupt_check(step)
            generated.append(token_id)
            inputs_embeds = self.llm.get_input_embeddings()(next_id.unsqueeze(0))
            attn = torch.cat([attn, torch.ones(1, 1, dtype=torch.long, device=self.device)], dim=1)

            if fire:
                t_interrupt = time.perf_counter()
                interrupted = True
                interrupt_to_halt_ms = (time.perf_counter() - t_interrupt) * 1000.0
                break
            if token_id == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(token_ids=generated, text=text, interrupted=interrupted,
                                 n_tokens_generated=len(generated), per_token_latencies_ms=latencies,
                                 interrupt_to_halt_ms=interrupt_to_halt_ms)
