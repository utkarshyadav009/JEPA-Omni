"""Jetson benchmark for Tier 1 speculative prefetch (SpeculativePrefetcher).
Loads ONLY the fast tier (LFM2 GGUF) + StreamingVoice -- no perception stack,
so no M3/NVML issue and no preflight needed. Measures, with the REAL models:
  - baseline response latency = fast-tier generate(final) + TTS time-to-first-audio
  - Tier-1 HIT latency        = commit() when the speculation (run during the
                                user's turn on the PARTIAL transcript) already
                                finished -> should be ~0
  - saved_ms per hit          = the gen+decode time removed from the response path
and verifies hit/miss correctness (a wrong guess must fall back, never commit).
"""
import sys, time, argparse, os
sys.path.insert(0, "/home/bmo/bmo_production/pipeline")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfm2", default="/home/bmo/bmo_production/models_gguf/bmo_lfm2_700m_v9_Q8_0.gguf")
    ap.add_argument("--tok", default="/home/bmo/bmo_production/tokenizers/lfm2_v3_tok")
    ap.add_argument("--tts", default="/home/bmo/bmo_production/models_gguf/bmo_neutts_v5_Q8_0.gguf")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from models.m4_cognitive_core import GGUFFastTier
    from models.m5_streaming_voice import StreamingVoice
    from models.m5_speculative import SpeculativePrefetcher

    print("loading fast tier (LFM2) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tok)
    fast = GGUFFastTier(args.lfm2, tok, max_new_tokens=24, n_gpu_layers=-1)
    print("loading StreamingVoice ...", flush=True)
    sv = StreamingVoice(args.tts, device=None)

    state = {"energy": 0.7, "mood": "content"}

    def gen_fn(transcript, st):
        return fast.generate(transcript, st)

    # --- component baselines (what a HIT hides) ---
    warm = fast.generate("hello beemo", state)  # warmup
    t0 = time.perf_counter(); r = fast.generate("what is the weather like today", state)
    gen_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter(); _ = next(sv.stream(r.text or "Okay!"), None)
    ttfa_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\nBASELINE components (Jetson): fast_tier_gen={gen_ms:.0f}ms  tts_first_audio={ttfa_ms:.0f}ms  "
          f"-> response path w/o Tier1 ~= {gen_ms+ttfa_ms:.0f}ms\n", flush=True)

    # --- scenarios: (partial seen mid-turn, final at end-of-turn, expect_hit) ---
    scenarios = [
        ("what is the weather like",        "what is the weather like today",   True),
        ("hey beemo how are you",           "hey beemo how are you doing",       True),
        ("beemo can you play a game",       "beemo can you play a game with me",  True),
        ("tell me a story about finn",      "what time is it right now",          False),
    ]
    pf = SpeculativePrefetcher(gen_fn, tts=sv, match_threshold=0.80, prewarm_audio=True)

    print(f"{'partial->final':52s} {'expect':6s} {'committed':9s} {'saved_ms':>8s} {'resp_ms':>7s}", flush=True)
    for partial, final, expect_hit in scenarios:
        # user is mid-turn: partial arrives -> speculate; user keeps talking.
        # Real turns leave 1-3s of remaining speech; simulate 1.5s so the
        # speculation (fast-tier gen + first-audio decode) can finish in the
        # background exactly as it would live. (If the user stopped instantly
        # after the partial, the spec wouldn't finish -> correct miss/fallback.)
        pf.speculate(partial, state)
        time.sleep(1.5)
        # end of turn: commit against the final transcript
        tc = time.perf_counter()
        result, first_audio, saved = pf.commit(final, state)
        commit_ms = (time.perf_counter() - tc) * 1000.0
        if result is not None:
            resp_ms = commit_ms  # instant: already computed
            committed = "HIT"
        else:
            # miss -> normal path cost now
            t0 = time.perf_counter(); rr = fast.generate(final, state); _ = next(sv.stream(rr.text or "Okay!"), None)
            resp_ms = (time.perf_counter() - t0) * 1000.0
            committed = "miss"
        ok = "OK" if (committed == "HIT") == expect_hit else "!! WRONG"
        tag = f"{partial[:24]} -> {final[:22]}"
        print(f"{tag:52s} {str(expect_hit):6s} {committed:9s} {saved:8.0f} {resp_ms:7.0f}  {ok}", flush=True)

    print(f"\nprefetcher stats: {pf.stats()}", flush=True)
    print("INTERPRETATION: on a HIT the response path is ~0ms (reply already computed "
          f"during the user's turn); ~{gen_ms+ttfa_ms:.0f}ms removed from perceived latency.", flush=True)

if __name__ == "__main__":
    main()
