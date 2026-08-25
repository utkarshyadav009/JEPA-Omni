# BMO emotion mapping — homeostatic mood → Fish tag → face expression

Unifies the three systems so one homeostatic state drives words (LLM, already wired),
voice emotion (via emotion-labeled fine-tune, data generated with these Fish tags), and
face (face_database.txt). All Fish tags below were VERIFIED accepted by the
`s2.1-pro-free` API against the community BMO voice (reference_id
`323847d4c5394c678e5909c2206725f6`), 2026-08-07 — every call succeeded.

| Homeostatic mood | Fish tag (verified) | Face expression (face_database.txt) |
|---|---|---|
| excited   | `[excited]`     | face_excited_stars |
| happy     | `[delight]`     | face_happy_standard / face_happy_talking |
| content   | `[relaxed]`     | face_happy_closed_eyes |
| surprised | `[surprised]`   | face_shocked_pale |
| stressed  | `[panting]`     | face_wincing |
| anxious   | `[nervous]`     | face_worried_teary |
| concerned | `[worried]`     | face_worried_teary |
| lonely    | `[sad]`         | face_sad_standard / face_sad_frown |
| tired     | `[sigh]`        | face_tired_droopy |
| bored     | `[indifferent]` | face_pixel_blank / face_pixel_annoyed |
| curious   | `[curious]`     | face_look_side / face_skeptical |

Notes:
- `homeostatic_to_mood_state()` (models/homeostatic_state.py) already emits these 11 mood
  labels; the face engine's AffectiveEngine already maps the appraisal vector to faces.
  This table adds the middle column (voice emotion) so all three stay congruent.
- The 30 face expressions cluster into ~11 emotion families; the mapping picks the closest
  family per mood. Some faces (love_hearts, kissy_face, smug, silly_tongue, dead_x,
  hypnotized) are special/interaction faces outside the homeostatic-mood set — reachable
  by other triggers, not this table.
- Fish also verified for non-verbal cues: `[whisper] [screaming] [laughing] [chuckle]
  [sigh] [panting] [inhale] [moaning]` (from scripts/generate_bmo_voice_corpus_fishapi.py,
  already working) — useful for the prebuilt backchannel bank too.

## Build path (after the base streaming pipeline is solid)
1. Generate emotion-labeled BMO audio via Fish (these tags) across the 11 moods.
2. Fine-tune NeuTTS with custom emotion tokens on it (keeps BMO's voice; TRAINING.md path).
3. Extend homeostatic_to_mood_state() (or a sibling) to also return the emotion token so
   the SAME state that already shapes the LLM's words shapes the voice delivery + face.
