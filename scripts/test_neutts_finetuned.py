"""scripts/test_neutts_finetuned.py -- real inference test of the
fine-tuned NeuTTS-Air checkpoint (checkpoints/bmo_neutts_finetune/best).

Not a zero-shot-cloning call (that's the `neutts` package's NeuTTS class,
which expects a packaged backbone repo) -- this is a raw HF causal-LM
generate() call replicating finetune.py's own chat format in reverse, since
our checkpoint is a plain Trainer output, not a packaged inference bundle.
"""
import re
import torch
import soundfile as sf
import phonemizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from neucodec import NeuCodec

CKPT = "checkpoints/bmo_neutts_finetune/best"
device = "cuda"

print("Loading fine-tuned NeuTTS-Air checkpoint...", flush=True)
# Real bug found: the fine-tune's Trainer wasn't given `tokenizer=`, so
# save_model() never wrote tokenizer files to the checkpoint dir --
# AutoTokenizer.from_pretrained silently fell back to a broken vocab_size=1
# tokenizer there (confirmed directly), producing empty input_ids and an
# IndexError deep in generate(). Fix: load the tokenizer from the original
# base repo -- the fine-tune only changes weights, not vocab/tokenization.
tokenizer = AutoTokenizer.from_pretrained("neuphonic/neutts-air")
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(device).eval()
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device).eval()

g2p = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True,
    words_mismatch="ignore", language_switch="remove-flags",
)

SPEECH_ID_RE = re.compile(r"<\|speech_(\d+)\|>")

test_lines = [
    "I feel lonely right now. Nothing happens in the empty house.",
    "Hello! I am functioning well today, thank you for asking!",
    "That was loud. My circuits got a little jumpy.",
]

for i, text in enumerate(test_lines):
    phones = " ".join(g2p.phonemize([text])[0].split())
    prompt = f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=800, do_sample=False,
            eos_token_id=speech_end_id, pad_token_id=tokenizer.pad_token_id,
        )
    gen_text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    speech_ids = [int(x) for x in SPEECH_ID_RE.findall(gen_text)]

    if not speech_ids:
        print(f"[{i}] NO SPEECH TOKENS GENERATED for: {text!r}", flush=True)
        continue

    codes = torch.tensor(speech_ids, dtype=torch.long, device=device).view(1, 1, -1)
    with torch.no_grad():
        wav = codec.decode_code(codes)
    out_path = f"/tmp/claude-1006/-home-utkarsh/dc0bf6a0-e1b8-4eb7-8ee4-798bf9178fb4/scratchpad/neutts_finetuned_{i}.wav"
    sf.write(out_path, wav.squeeze().float().cpu().numpy(), 16000)
    print(f"[{i}] {len(speech_ids)} speech tokens -> {out_path}  text={text!r}", flush=True)

print("DONE", flush=True)
