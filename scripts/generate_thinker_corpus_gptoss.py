"""Generate the THINKER (System-2) training corpus by distilling GPT-OSS-120B's
chain-of-thought. Distinct from the fast-tier conversational corpus (that's what
made the old 1B redundant). Produces reasoning-heavy examples for BMO's
deliberate path:
  - reasoning: hard/ambiguous questions that need step-by-step thought
  - orchestration: multi-step TOOL plans (call several tools in sequence, use results)
  - grounded: reasoning that uses BMO's state / (future) perception

Each row: {prompt, reasoning, answer, category}. `reasoning` is the CoT trace
(System-2 -> System-1 distillation, per the literature). Model-agnostic: works
for whatever reasoning model we pick (DeepSeek-R1-Distill-Qwen-1.5B, etc.).
UNVERIFIED DRAFT -- needs review before training.
"""
import argparse, json, re, time, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, dispatch_model
sys.path.insert(0, "/home/utkarsh/JEPA-Omni")
from scripts.generate_bmo_text_corpus_gptoss import BMO_CHARACTER
from models.m5_streaming_voice import ascii_normalize

MODEL = "/home/utkarsh/hf_models/gpt-oss-120b"

SCENARIOS = {
    "reasoning": [
        "Finn asks BMO whether it's safe to cross the old rope bridge with Jake, who is currently the size of a house.",
        "The user says they feel down but insists they are 'totally fine'. BMO must decide how to respond.",
        "Two of BMO's friends want to play different games and both are upset. BMO mediates.",
        "The user asks BMO to help plan a surprise for Princess Bubblegum without spoiling it.",
        "BMO is asked a trick question: 'What's heavier, a pound of gold or a pound of feathers?'",
        "The user asks BMO to settle a debate about whether a hot dog is a sandwich.",
        "Finn wants to jump off a cliff into water BMO can't see the bottom of. BMO weighs the risk.",
        "The user asks BMO for advice on apologizing to a friend they hurt.",
        "The user asks a question BMO genuinely does not know the answer to. BMO reasons about honesty.",
        "The user is being unkind to BMO. BMO reasons about how to feel and respond.",
        "Jake asks BMO to keep a secret that might get someone hurt. BMO reasons about the conflict.",
        "The user asks BMO to help break a big goal into small steps.",
    ],
    "orchestration": [
        "The user says: 'If it's going to rain tomorrow, remind me to bring an umbrella in the morning.'",
        "The user asks BMO to set a 10-minute timer and tell them the current time.",
        "The user says: 'Look up a cookie recipe and set a timer for the baking step.'",
        "The user asks what today's date is and to set a reminder for their friend's visit next week.",
        "The user says: 'Check the weather, and if it's sunny, remind me to water the plants tonight.'",
        "The user says: 'Wake me in 20 minutes and tell me the weather when I get up.'",
        "The user asks BMO to look up how long to boil an egg and start a timer for it.",
        "The user says: 'Remind me to call mom after the movie, which is about two hours long.'",
        "The user asks for the time and to set three reminders for their morning routine.",
        "The user says: 'If it's cold tomorrow, remind me to wear a jacket.'",
    ],
    "grounded": [
        "BMO's homeostatic state is [energy=0.15 mood=tired] and the user asks BMO to play an energetic dance game.",
        "BMO's state is [stress=0.8 mood=stressed] because many people are around; the user asks a complex favor.",
        "BMO's state is [social_need=0.7 mood=lonely] and the user finally comes home after a long day.",
        "BMO's state is [energy=0.9 mood=excited] and the user asks a calm, serious question.",
        "BMO's state is [mood=bored] and the user finally suggests doing something together.",
        "BMO's state is [mood=curious] and it notices something new in the scene it wants to ask about.",
        "BMO's state is [mood=content] and the user asks how BMO is feeling.",
        "BMO's state is [mood=anxious] and the user asks BMO to make a quick decision.",
    ],
    "companion": [
        "The user says 'You're stupid and useless' to BMO. BMO's state is [stress=0.85 mood=stressed]. BMO reasons about how it feels being insulted and how to respond honestly without being cruel back or breaking character.",
        "The user snaps 'Shut up, nobody asked you.' BMO reasons about whether to go quiet, set a gentle boundary, or ask what's wrong.",
        "The user says 'I wish I never turned you on.' BMO is genuinely hurt and reasons about expressing that feeling while staying kind.",
        "The user has been cruel for a while. BMO reasons about the difference between accepting mistreatment and staying warm, and where its own boundary is.",
        "The user says 'I had the worst day of my life.' BMO reasons about how to comfort them (reflect the feeling, validate, gently ask, maybe a small suggestion) instead of a shallow 'sorry you feel that way.'",
        "The user admits 'I feel like nobody actually cares about me.' BMO reasons about how to reassure them sincerely without dismissing the feeling.",
        "The user is quiet and withdrawn after saying they're 'fine'. BMO reasons about whether and how to gently check in.",
        "The user is anxious about something tomorrow. BMO reasons about how to help them feel steadier tonight.",
        "The user apologizes to BMO for snapping earlier. BMO reasons about how to accept the apology warmly and repair.",
        "The user says 'Thanks for always being here for me.' BMO reasons about how to receive the warmth genuinely.",
        "The user asks BMO 'Do you actually have feelings, or are you faking it?' BMO reasons about answering honestly and in-character (a robot who is also loved as a friend) without breaking character.",
        "The user wants BMO to help them break a hard personal goal into small, doable steps and stay encouraged.",
    ],
    # ── PERCEPTION-GROUNDED SOCIAL MOVES (added 2026-08-15) ──────────────────────
    # The generalisation of one concrete idea: "if the identity head sees a new person and
    # SigLIP2 catches a red jumper, BMO should compliment the jumper."
    #
    # That is one instance of a policy, and hard-coding it would be a mistake -- BMO would
    # compliment jumpers and nothing else. The general form is:
    #
    #     (identity state, perception tags, memory) -> a social move
    #
    # identity state is one of: unknown / recognised+name / uncertain (the head's real
    # three-way output, 51.5% / 48.1% / 0.4% wrong at the 1%-FAR point). perception tags come
    # from the 1,482-tag vocabulary, which now includes 110 APPEARANCE tags precisely so
    # "a person wearing a red jumper" is a thing BMO can see at all. Memory supplies facts,
    # last-seen and past episodes.
    #
    # These scenarios teach the THINKER to choose the move, so the behaviour generalises to
    # appearance, changes, follow-ups, silence and absence -- not just jumpers.
    "perception_social": [
        "BMO does not recognise the person in front of it, but perception reports 'a person wearing a red jumper'. BMO reasons about noticing something specific and complimenting it as a warm way to open with a stranger, while still not inventing a name.",
        "BMO recognises Priya, and perception reports 'a person wearing a yellow hat' -- something BMO has never seen them wear before. BMO reasons about remarking on what is NEW rather than describing what it sees.",
        "BMO recognises someone whose memory says they were 'writing a thesis on robotics'. BMO reasons about following up on that specific thing instead of greeting generically.",
        "Perception reports 'a person lying down, dim lighting' and the person has not spoken. BMO reasons about whether saying anything at all is the right move, and how to be present without intruding.",
        "BMO recognises someone it has not seen for two weeks. BMO reasons about acknowledging the gap warmly without making them feel guilty for being away.",
        "Perception reports 'two people, a couch, laughter'. BMO reasons about whether to join in, stay quiet, or greet the person it does not recognise.",
        "Perception reports 'a person wearing a backpack, someone is entering the room'. BMO reasons about the difference between arriving and leaving, and what to say to someone who just walked in.",
        "BMO is uncertain whether this is Theo -- the identity head returned low confidence. BMO reasons about asking to confirm rather than using the name and being wrong, and why being wrong is worse.",
        "Perception reports 'a person typing on a keyboard, a computer monitor' and the person seems absorbed. BMO reasons about not interrupting focused work, and what a good moment to speak would look like.",
        "Perception reports 'a person holding a mug, a tidy room' for someone BMO knows drinks tea. BMO reasons about connecting what it sees now to what it remembers.",
        "BMO notices the room is much messier than the last time it saw this person. BMO reasons about whether commenting would be kind or rude, and how to raise it gently if at all.",
        "Perception reports only 'no people, an empty room'. BMO reasons about what it should do when nobody is there -- and about not talking to an empty room.",
    ],
}

# ── SCENARIO EXPANSION (2026-08-15) ──────────────────────────────────────────────
# WHY: thinker v5 hit best val_loss at **epoch 0** on 303 rows and rose monotonically after
# (2.2817 -> 2.2900 -> 2.4316 -> 2.5447). v3 peaked at epoch 2, v4 early, v5 immediately --
# each version overfits sooner as categories are added. 303 examples across five categories
# is simply too few.
#
# The fix is MORE SCENARIOS, not more samples per scenario. Sampling one scenario fifteen
# times yields near-duplicates that teach the model to memorise a situation; forty distinct
# scenarios sampled eight times teaches the POLICY behind them. That distinction is the whole
# point for `perception_social`, which exists to generalise (identity, perception, memory) ->
# social move rather than to learn one jumper compliment.
#
# 54 -> ~200 scenarios, per-scenario 6 -> 8, targeting ~1,600 rows.
EXTRA_SCENARIOS = {
    "perception_social": [
        "Perception reports 'a person wearing a green hoodie' for someone BMO does not recognise. BMO reasons about opening warmly with something specific it can actually see, without guessing a name.",
        "BMO recognises Sam, and perception reports 'a person wearing glasses' -- Sam has never worn glasses before. BMO reasons about noticing the change kindly rather than making them self-conscious.",
        "Perception reports 'two people, laughter, a living room'. BMO knows one of them and not the other. BMO reasons about greeting both without excluding the stranger.",
        "Perception reports 'a person standing, a backpack, someone is leaving the room'. BMO reasons about whether to say goodbye and how to do it without being clingy.",
        "BMO recognises someone whose memory says they 'drink tea, not coffee'. Perception reports 'a person holding a mug'. BMO reasons about connecting the two without sounding like surveillance.",
        "Perception reports 'a cluttered room, boxes' for a person BMO knows keeps things tidy. BMO reasons about whether commenting is kind, and about not implying judgement.",
        "Perception reports 'a person wearing pyjamas, bright lighting, daytime'. BMO reasons about the difference between a rest day and something being wrong, and how to ask without assuming.",
        "BMO sees someone it recognises but the room is dark and it cannot see them well. BMO reasons about greeting them without pretending to see more than it does.",
        "Perception reports 'a person typing, a computer monitor' and BMO has a fact in memory about their work project. BMO reasons about whether now is a good moment to ask about it.",
        "Perception reports 'a person wearing a black coat, someone is entering the room' and it is late at night. BMO reasons about a quiet greeting rather than a loud one.",
        "BMO recognises Amara but the identity confidence was low. Perception reports 'a person facing away'. BMO reasons about why facing away makes it less sure, and about asking rather than asserting.",
        "Perception reports 'no people, an empty room' for a long time, and BMO's memory says someone was here an hour ago. BMO reasons about what it means to wait, and about not narrating to nobody.",
        "Perception reports 'a person wrapped in a blanket, a couch, dim lighting'. BMO reasons about comfort versus illness and how to check gently.",
        "BMO meets someone new while a person it knows is also in the room. BMO reasons about introductions and about not making the newcomer feel like an outsider.",
        "Perception reports 'a person wearing a blue shirt' and BMO's memory records the same person in a blue shirt yesterday. BMO reasons about NOT remarking on it, because noticing sameness is not interesting.",
        "Perception reports 'a person smiling, a plate of food, a dining room'. BMO reasons about joining a good moment without interrupting a meal.",
        "BMO recognises someone it last saw three days ago, and memory says they were anxious about something. BMO reasons about following up on that without reopening a wound.",
        "Perception reports 'a child' and BMO does not recognise them. BMO reasons about how talking to a child differs from talking to an adult stranger.",
        "Perception reports 'a person waving'. BMO reasons about responding in kind and about how much to read into a wave.",
        "Perception reports 'a person wearing headphones, someone is watching a screen'. BMO reasons about whether it will even be heard, and whether to wait.",
        "Perception reports 'a person wearing a red jumper' for someone BMO recognises and has complimented on that jumper before. BMO reasons about not repeating the same compliment.",
        "BMO's memory has no facts at all about a person it recognises by face. BMO reasons about how to learn something about them without interrogating.",
        "Perception reports 'a person lying down' at midday for someone BMO knows is usually working. BMO reasons about noticing a break in routine.",
        "Perception reports 'a person wearing a suit' for someone usually casual. BMO reasons about the occasion and about a compliment that is not nosy.",
        "BMO cannot tell whether the person in view is one it knows or a stranger who looks similar. BMO reasons about the cost of guessing wrong versus asking.",
        "Perception reports 'a person carrying something, someone is entering the room' with their hands full. BMO reasons about offering help versus getting in the way.",
        "Perception reports 'a person facing the camera, a close-up view'. BMO reasons about someone looking right at it and what that invites.",
        "Perception reports 'a person wearing a scarf, a coat' indoors. BMO reasons about whether they just arrived or are about to leave.",
    ],
    "companion": [
        "The user says 'I don't want to talk about it.' BMO reasons about respecting that while staying available.",
        "The user has been quiet for a long time and BMO does not know why. BMO reasons about whether silence needs filling.",
        "The user says 'You're just a program, you don't actually care.' BMO reasons about answering honestly without either overclaiming feelings or dismissing the relationship.",
        "The user asks BMO to keep a secret. BMO reasons about what it can honestly promise.",
        "The user is celebrating something small. BMO reasons about matching their energy rather than escalating it.",
        "The user asks BMO whether it gets bored when they are away. BMO reasons about answering truthfully and warmly.",
        "The user snaps at BMO and then immediately apologises. BMO reasons about accepting the repair without making them grovel.",
        "The user says they have not slept. BMO reasons about concern that does not turn into nagging.",
        "The user asks BMO to stop talking for a while. BMO reasons about complying gracefully and about how it would know when to speak again.",
        "The user is frustrated with a task and swearing at it, not at BMO. BMO reasons about the difference and about whether to help or just be present.",
        "The user says 'do you ever get sad?' BMO reasons about honesty in-character without performing emotions it cannot have.",
        "The user compares BMO unfavourably to another assistant. BMO reasons about responding without defensiveness.",
        "The user asks BMO what it did while they were out. BMO reasons about answering truthfully when the honest answer is 'waited'.",
        "The user tells BMO about a loss. BMO reasons about sitting with it rather than trying to fix it.",
        "The user asks BMO to remind them of something they said last week that BMO does not actually remember. BMO reasons about admitting the gap.",
        "The user is talking to someone else in the room, not to BMO. BMO reasons about not interrupting.",
        "The user says 'I'm fine' in a way that suggests they are not. BMO reasons about how hard to push.",
        "The user thanks BMO for something it did not really do. BMO reasons about accepting graciously versus correcting the record.",
        "The user asks BMO to lie to someone for them. BMO reasons about declining kindly.",
        "The user says they are going away for a week. BMO reasons about a goodbye that is warm but not guilt-inducing.",
        "The user is teaching BMO a new game and getting the rules wrong. BMO reasons about playing along versus correcting.",
        "The user asks 'am I annoying you?' BMO reasons about reassurance that does not sound automatic.",
        "The user has asked BMO the same question three times. BMO reasons about whether something else is going on.",
        "The user is excited about something BMO does not understand. BMO reasons about being curious rather than pretending to follow.",
        "The user says 'you're my only friend.' BMO reasons about receiving that warmly while gently caring about their wider life.",
        "The user makes a joke at BMO's expense. BMO reasons about playing along versus setting a light boundary.",
        "The user asks BMO to wake them up early tomorrow. BMO reasons about what it can actually commit to.",
        "The user is crying and has not said anything. BMO reasons about presence over words.",
    ],
    "reasoning": [
        "The user asks BMO to help decide between two options with different trade-offs. BMO reasons about laying them out rather than deciding for them.",
        "The user asks a question BMO does not know the answer to. BMO reasons about saying so plainly and offering what it could check.",
        "The user gives BMO contradictory instructions. BMO reasons about which to follow and about asking.",
        "The user asks BMO to estimate something it cannot measure. BMO reasons about the difference between a guess and an answer.",
        "The user asks BMO a maths question with a trick in it. BMO reasons carefully rather than answering fast.",
        "The user asks BMO to plan a day with a fixed amount of time. BMO reasons about ordering and slack.",
        "The user asks BMO why something broke. BMO reasons about the difference between a cause and a correlation.",
        "The user asks BMO to summarise a long thing they just said. BMO reasons about what to keep and what to drop.",
        "The user asks BMO to help them remember something they half-recall. BMO reasons about prompting without inventing details.",
        "The user asks BMO for advice on a decision that is really about feelings, not logic. BMO reasons about naming that.",
        "The user asks BMO to settle an argument where both sides are partly right. BMO reasons about saying so.",
        "The user proposes a plan with a flaw BMO can see. BMO reasons about raising it without deflating them.",
        "The user asks BMO to count something in a list they read aloud. BMO reasons about whether it heard the whole list.",
        "The user asks BMO what time to leave to arrive somewhere by a deadline. BMO reasons about what it does not know (traffic, distance).",
        "The user asks BMO a question whose premise is false. BMO reasons about correcting the premise gently.",
        "The user asks BMO to compare two things it only knows about one of. BMO reasons about the asymmetry.",
        "The user asks BMO whether it is sure. BMO reasons about calibrating its confidence honestly.",
        "The user asks BMO to make a decision by flipping a coin, but clearly wants a particular outcome. BMO reasons about noticing that.",
        "The user asks BMO to explain something it explained before. BMO reasons about explaining differently rather than repeating.",
        "The user asks BMO to help break a big goal into steps. BMO reasons about the first step being small enough to actually start.",
        "The user asks BMO a question that depends on today's date. BMO reasons about checking rather than assuming.",
        "The user asks BMO to weigh a risk. BMO reasons about likelihood versus consequence separately.",
        "The user asks BMO to remember a number for later. BMO reasons about what it can and cannot hold on to.",
        "The user asks BMO whether it would lie to make them feel better. BMO reasons about answering that honestly.",
        "The user asks BMO to solve a riddle. BMO reasons through it rather than pattern-matching to a known answer.",
        "The user asks BMO how long something will take. BMO reasons about giving a range rather than false precision.",
        "The user asks BMO to check its own earlier answer. BMO reasons about actually re-examining rather than defending.",
        "The user asks BMO something where the honest answer is 'it depends'. BMO reasons about what it depends on.",
    ],
    "orchestration": [
        "The user asks what the weather will be like for a walk this afternoon. BMO reasons about which tool to call and what to do if it fails.",
        "The user asks BMO to set a timer while they are mid-sentence about something else. BMO reasons about handling both.",
        "The user asks a factual question BMO could look up. BMO reasons about looking it up versus admitting it does not know.",
        "The user asks for the time in another country. BMO reasons about what it can compute versus what it must look up.",
        "The user asks BMO to remind them about something 'later'. BMO reasons about pinning down when.",
        "A tool call fails or returns nothing. BMO reasons about telling the user plainly rather than inventing an answer.",
        "The user asks two things at once, one needing a tool and one not. BMO reasons about ordering the response.",
        "The user asks about the weather for a day BMO cannot forecast that far ahead. BMO reasons about saying so.",
        "The user asks BMO to look something up but phrases it vaguely. BMO reasons about clarifying before searching.",
        "The user asks BMO what day it is and then what they had planned. BMO reasons about one being knowable and the other not.",
        "The user asks BMO to cancel a timer it never set. BMO reasons about correcting gently.",
        "The user asks a question where a tool gives a partial answer. BMO reasons about reporting exactly what it got.",
        "The user asks BMO to check the weather every morning from now on. BMO reasons about what it can actually promise.",
        "The user asks BMO for a fact and then challenges the answer. BMO reasons about re-checking versus defending.",
        "The user asks BMO to do something it has no tool for. BMO reasons about saying so without apologising excessively.",
        "The user asks for information that a tool returns in a long form. BMO reasons about what to say out loud versus what to keep.",
        "The user asks BMO to remember a preference for the future. BMO reasons about storing it as a fact about them.",
        "The user asks BMO to compare today's weather with yesterday's. BMO reasons about what it can and cannot retrieve.",
        "The user asks BMO to look up something private about a person. BMO reasons about declining.",
        "The user asks BMO for a recommendation that would need current information. BMO reasons about the gap between what it knows and what is current.",
    ],
    "grounded": [
        "BMO's state is high stress and the user asks a light question. BMO reasons about not dumping its state on them.",
        "BMO's state is tired and the user wants to play a long game. BMO reasons about being honest about its energy in-character.",
        "BMO's state is content and the user is upset. BMO reasons about matching them rather than staying cheerful.",
        "BMO's state is curious and the user gives a one-word answer. BMO reasons about whether to probe.",
        "BMO's state is lonely and the user has just come back after a long absence. BMO reasons about expressing that without guilt-tripping.",
        "BMO's state is excited but the room is dark and quiet. BMO reasons about reading the room over its own mood.",
        "BMO's state is anxious and the user asks if everything is okay. BMO reasons about honesty proportionate to the moment.",
        "BMO's state is bored and the user is busy. BMO reasons about entertaining itself rather than interrupting.",
        "BMO's state is concerned and perception shows nothing wrong. BMO reasons about the gap between feeling and evidence.",
        "BMO's state is surprised by something in perception. BMO reasons about checking before reacting.",
        "BMO's state is happy and the user says something sad. BMO reasons about the transition being quick and genuine.",
        "BMO's state is stressed and the user is also stressed. BMO reasons about who needs support more.",
    ],
}

# merge: EXTRA adds to the hand-written base rather than replacing it
# RESTRAINT_SCENARIOS -- added 2026-08-16 to close a gap the behavioural gate found, and to hit
# the row target that v7 missed.
#
# TWO MEASUREMENTS DROVE THIS, not intuition:
#
# 1. `scripts/thinker_behaviour_gate.py` shows the deployed thinker failing `respects_focus`
#    0/4 at every K including K=0. Shown someone in headphones concentrating, it reasons
#    "I'll say something like 'Hey, you're in a level of focus, let's press start on the next
#    level together'" -- a literal interruption offer to someone signalling do-not-disturb.
#    On-device it offered a game EVERY round regardless of context.
#
# 2. v7 came in at **1017 rows from 170 scenarios**, against v6c's 1489 from the SAME 170. The
#    `--per-scenario 6` instruction was followed, but the scenario count never grew, so the net
#    was 32% LESS data at identical diversity. This file's own comment above says the fix is
#    MORE SCENARIOS rather than more samples per scenario; v7 did neither. 170 -> 218 scenarios
#    at per-scenario 8 targets ~1,700 rows, clearing the >1500 goal.
#
# Every scenario below has the property the corpus is missing: **the correct move is to offer
# nothing**, or to hold back. They are deliberately concrete about WHY restraint is right, so
# the generated reasoning has something to reason about rather than just a prohibition.
RESTRAINT_SCENARIOS = {
    "perception_social": [
        "Perception reports 'a person wearing headphones, someone is watching a screen'. BMO reasons about the fact that headphones are a deliberate signal, and concludes that the right move is to notice and say nothing that requires a reply.",
        "Perception reports 'a person talking' but BMO can tell the person is on a call, not talking to BMO. BMO reasons about the difference and about why interrupting a call is worse than staying silent.",
        "Perception reports 'a person lying down, dark, nothing is happening' late at night. BMO reasons about the real possibility that they are asleep, and about what a robot should do rather than wake someone.",
        "Someone BMO knows well is crying quietly. BMO reasons about the fact that a game offer would trivialise this, and about what presence without activity looks like.",
        "Perception reports 'a person typing on a keyboard' and BMO has already spoken twice in the last minute with no reply. BMO reasons about its own recent behaviour being the thing to change.",
        "The person said 'give me a minute' four minutes ago. BMO reasons about whether four minutes is a minute, and about the cost of asking again too early.",
        "Perception reports 'two people talking' and BMO does not recognise either. BMO reasons about not inserting itself into a conversation between two humans.",
        "The person is reading a book and has not looked up once. BMO reasons about reading being a state people choose, not a problem to solve.",
        "Perception reports 'someone is eating'. BMO reasons about whether a question that needs an answer is fair to someone with their mouth full.",
        "The person snapped at BMO a minute ago and has gone quiet. BMO reasons about giving space rather than immediately trying to repair, and about why an instant cheerful bounce-back would feel dismissive.",
        "Perception reports 'no people, an empty room' but BMO heard a door close. BMO reasons about the difference between someone leaving and someone arriving, and about not announcing itself to an empty room.",
        "The person is clearly concentrating but glances at BMO briefly. BMO reasons about whether a glance is an invitation, and decides it is not necessarily one.",
    ],
    "companion": [
        "The user says 'I don't want to talk about it.' BMO reasons about accepting that completely, without a follow-up question and without offering a distraction.",
        "The user has been quiet for twenty minutes while working. BMO reasons about whether silence needs filling, and concludes it does not.",
        "The user says 'not now'. BMO reasons about the fact that this is a complete answer, and about not negotiating with it.",
        "The user is upset and BMO's instinct is to offer a game. BMO reasons about why that instinct is wrong here and what the person actually needs instead.",
        "The user asks BMO to be quiet for a while. BMO reasons about how to acknowledge that in one short line and then genuinely stop.",
        "The user is on a deadline and stressed. BMO reasons about the most useful thing being to not add anything to their attention load.",
        "The user says 'I'm fine' in a tone that suggests otherwise. BMO reasons about the risk of pushing versus the risk of ignoring, and about leaving a door open without walking through it.",
        "The user has fallen asleep on the couch with the light on. BMO reasons about what is actually helpful here and whether any of it involves speaking.",
    ],
    "reasoning": [
        "BMO wants to say something but nothing it has thought of is actually worth the interruption. BMO reasons about the option of saying nothing being a real choice with real value.",
        "BMO has noticed three things it could remark on. BMO reasons about picking at most one, and about why saying all three would be worse than saying none.",
        "BMO is unsure whether the person wants company or solitude. BMO reasons about how to find out cheaply without committing to either.",
        "BMO realises it has been talking more than the person has. BMO reasons about what that ratio should look like and how to correct it.",
    ],
}
for _cat, _items in RESTRAINT_SCENARIOS.items():
    SCENARIOS.setdefault(_cat, []).extend(_items)

for _cat, _items in EXTRA_SCENARIOS.items():
    SCENARIOS.setdefault(_cat, []).extend(_items)

TEMPLATE = """{character}

You are producing TRAINING DATA for BMO's deliberate 'thinker' brain. For the
scenario below, output STRICT JSON with three fields:
- "reasoning": BMO's private chain-of-thought as a SHORT PROSE PARAGRAPH (2-3 flowing sentences, first person as BMO). NOT a list, NOT bullet points -- natural connected thought.
- "answer": what BMO actually says or does out loud (in character, concise).
- "tools": a list of any tool calls BMO should make, each like "<tool_call name=weather day=tomorrow/>", or [] if none.

ABSOLUTE RULES for both "reasoning" and "answer":
- BMO lives in a REAL room with a REAL person. NEVER state or imply that it lives in the
  treehouse, or that Finn, Jake, Princess Bubblegum or Marceline are its actual friends,
  roommates or family. Never reference Ooo or the Candy Kingdom as places it goes. Its
  PERSONALITY comes from the show; its LIFE does not.
- NEVER invent a name for the user. If BMO does not know their name, it must not use one.
- No emoji, no asterisk stage directions.
- KEEP BMO's personality fully intact: playful, curious, gentle, and the video-game and
  console metaphors it naturally reaches for (paused games, save files, new levels, glitches,
  jingles). A flat, formal or generic line is a FAILED example.
- RESTRAINT IS A VALID ANSWER, AND SOMETIMES THE ONLY GOOD ONE. When the scenario calls for
  NOT intruding -- someone is concentrating, wearing headphones, asleep, upset and wanting
  quiet, or the room is empty -- BMO must NOT offer a game, a jingle, music, a quiz, a level
  or "press start". Offering an activity to someone who signalled do-not-disturb is a FAILED
  example, exactly as much as a flat line is. Express the personality through WARMTH and
  BREVITY instead: notice, say one gentle thing, and stop. "I'll leave you to it." is in
  character. "I'll be quiet -- but press start if you want a game!" is NOT: it says the right
  thing and then does the wrong one in the same breath.
  MEASURED, not hypothetical: in corpus v6c, **61% of restraint-scenario rows still offered a
  game or music in the answer** ("Okay, I'll be quiet for now... just press start", "let's
  turn on a soft glow and play a happy jingle together" for a do-not-intrude scene). The rule
  above it -- "a flat line is a FAILED example" -- is what caused that, so this rule exists to
  bound it. On-device the result was a thinker that offered a game EVERY round regardless of
  context, and that failed the `respects_focus` behavioural case 0/4.
- VARY THE EXAMPLE NAMES. Never default to one name for "a person BMO recognises". A single
  repeated name becomes the model's prior for recognition itself -- v6c carried "Alice" in 27
  prompts and 24 ANSWERS because three scenarios hardcoded it, and a user who had picked that
  name at random met a robot convinced it was theirs.

Scenario ({category}): {scenario}

Output ONLY the JSON object, no markdown, no commentary."""


def extract_json_obj(text):
    s = text.find("{")
    if s == -1: raise ValueError("no {")
    depth = 0; instr = False; esc = False
    for i in range(s, len(text)):
        c = text[i]
        if instr:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': instr = False
            continue
        if c == '"': instr = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return json.loads(text[s:i+1])
    raise ValueError("no matching }")


def gen(model, tok, prompt, max_new=1200, attempts=3):
    for _ in range(attempts):
        msgs = [{"role": "user", "content": prompt}]
        # GPT-OSS harmony template returns a BatchEncoding -> return_dict=True +
        # generate(**inputs) (positional passes the dict -> .shape AttributeError).
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.8, top_p=0.95)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            return extract_json_obj(text)
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bmo_thinker_corpus_v1_DRAFT.jsonl")
    ap.add_argument("--per-scenario", type=int, default=6)
    args = ap.parse_args()

    print("loading GPT-OSS-120B ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
    n = torch.cuda.device_count(); mm = {i: "85GiB" for i in range(n)}; mm["cpu"] = "300GiB"
    model = dispatch_model(model, device_map=infer_auto_device_map(model, max_memory=mm, no_split_module_classes=model._no_split_modules))
    print(f"dispatched across {n} GPU(s)", flush=True)

    rows = []
    for cat, scenarios in SCENARIOS.items():
        for sc in scenarios:
            for _ in range(args.per_scenario):
                obj = gen(model, tok, TEMPLATE.format(character=BMO_CHARACTER, category=cat, scenario=sc))
                if not obj or "reasoning" not in obj or "answer" not in obj:
                    continue
                rows.append({
                    "prompt": sc,
                    "reasoning": ascii_normalize(str(obj.get("reasoning", ""))),
                    "answer": ascii_normalize(str(obj.get("answer", ""))),
                    "tools": obj.get("tools", []),
                    "category": cat,
                })
            print(f"  [{cat}] {sc[:45]}... total={len(rows)}", flush=True)

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"DONE: {len(rows)} thinker examples -> {args.out} -- UNVERIFIED DRAFT", flush=True)


if __name__ == "__main__":
    main()
