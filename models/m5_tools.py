"""models/m5_tools.py -- the tool-call execution layer BMO was missing.

The LLM was fine-tuned to EMIT structured calls like
`<tool_call name=weather day=today/>` (see the corpus' tool_use slice), but
nothing in the runtime ever parsed or executed them -- so tool use was
non-functional end to end. This module closes that gap:

  1. parse_tool_calls(text)  -> [(name, params), ...] from the LLM output
  2. ToolRegistry           -> name -> handler(params) -> short result string
  3. ToolDispatcher.resolve -> run the tools and fold their results back into
     BMO's spoken line (template mode, reliable; or an optional LLM second
     pass once the model is trained to consume [tool_result ...] context).

Handlers that can be real ARE real (time/date). The rest (weather/search) are
clean stubs with an obvious injection point for a real API/key -- the mechanism
is what unlocks tool use; swapping a stub for a live call is a one-liner.

Pairs with a GBNF grammar (assets/tool_call.gbnf) that can constrain llama.cpp
decoding to only-valid tool-call syntax, so parsing never fails on a malformed
tag -- fine-tuning teaches WHEN/WHAT to call, the grammar guarantees the FORM.
"""
from __future__ import annotations

import re
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# <tool_call name=weather day=today/>  (self-closing, space-separated key=value)
_TOOL_CALL_RE = re.compile(r"<tool_call\s+([^>]*?)/?>", re.IGNORECASE)
_ATTR_RE = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|(\S+))")


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, str]]]:
    """Extract [(name, {param: value})] from any <tool_call .../> tags."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text or ""):
        attrs = {}
        for a in _ATTR_RE.finditer(m.group(1)):
            key = a.group(1)
            val = a.group(2) or a.group(3) or a.group(4) or ""
            attrs[key.lower()] = val
        name = attrs.pop("name", "").lower()
        if name:
            calls.append((name, attrs))
    return calls


def strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text or "").strip()


# ---- default handlers -------------------------------------------------------
def _h_time(p):
    return datetime.now().strftime("%-I:%M %p").lstrip("0")

def _h_date(p):
    return datetime.now().strftime("%A, %B %-d")

# Read once at import; None means "no key configured" -> the handler refuses rather than
# inventing. Kept OUT of the repo deliberately (same pattern as the Fish Audio key).
_OWM_KEY_PATH = os.path.expanduser("~/.config/bmo/openweather_api_key")
_OWM_KEY = None
if os.path.exists(_OWM_KEY_PATH):
    try:
        with open(_OWM_KEY_PATH) as _f:
            _OWM_KEY = _f.read().strip() or None
    except OSError:
        _OWM_KEY = None

# Where BMO lives. Overridable via ~/.config/bmo/location as "lat,lon,Name".
_OWM_LOC = (51.5074, -0.1278, "here")
_LOC_PATH = os.path.expanduser("~/.config/bmo/location")
if os.path.exists(_LOC_PATH):
    try:
        _parts = open(_LOC_PATH).read().strip().split(",")
        _OWM_LOC = (float(_parts[0]), float(_parts[1]),
                    _parts[2].strip() if len(_parts) > 2 else "here")
    except Exception:
        pass


def _h_weather(p):
    """Real OpenWeatherMap call. Returns None when it cannot answer.

    THE OLD STUB RETURNED "sunny and about seventy-two degrees" UNCONDITIONALLY. That is the
    same failure class this project spent a day removing from the identity path: confidently
    stating something it does not know. A companion that invents the weather during a
    thunderstorm is worse than one that says it cannot check.

    Returning None lets ToolDispatcher fold in an honest "I can't check that right now"
    instead of a fabrication. Short timeout because this sits on a ~1.8 s conversational
    tick -- a slow answer is a failed answer here."""
    if not _OWM_KEY:
        return None
    lat, lon, place = _OWM_LOC
    day = (p.get("day") or "today").lower()
    try:
        import json as _json
        from urllib.parse import urlencode
        from urllib.request import urlopen
        if day in ("today", "now", "currently", ""):
            url = ("https://api.openweathermap.org/data/2.5/weather?"
                   + urlencode({"lat": lat, "lon": lon, "appid": _OWM_KEY, "units": "metric"}))
            with urlopen(url, timeout=3.0) as r:
                d = _json.load(r)
            desc = d["weather"][0]["description"]
            temp = round(d["main"]["temp"])
            return f"{desc}, about {temp} degrees"
        # forecast: 3-hourly; take the entry ~24h out for "tomorrow"
        url = ("https://api.openweathermap.org/data/2.5/forecast?"
               + urlencode({"lat": lat, "lon": lon, "appid": _OWM_KEY,
                            "units": "metric", "cnt": 9}))
        with urlopen(url, timeout=3.0) as r:
            d = _json.load(r)
        e = d["list"][-1]
        return f"{e['weather'][0]['description']}, about {round(e['main']['temp'])} degrees {day}"
    except Exception:
        return None        # network down, bad key, rate limit -> say nothing rather than lie

def _h_timer(p):
    dur = p.get("duration") or p.get("minutes") or p.get("for") or "a few minutes"
    return f"a timer for {dur}"

def _h_reminder(p):
    what = p.get("text") or p.get("what") or p.get("about") or "that"
    when = p.get("when") or p.get("time") or ""
    return f"a reminder to {what}{(' ' + when) if when else ''}"

# BMO's whole context is 512 tokens and the real prompt is ~34 of them. A tool result that
# returns a paragraph would evict the scene description BMO needs to stay grounded, so both
# retrieval tools below hard-cap what they hand back. Short and true beats complete.
_TOOL_MAX_CHARS = 240


def _clip(text: str, limit: int = _TOOL_MAX_CHARS) -> str:
    """Trim to a sentence boundary inside the limit, never mid-word."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    for stop in (". ", "! ", "? "):
        i = cut.rfind(stop)
        if i > limit * 0.5:
            return cut[:i + 1].strip()
    return cut[:cut.rfind(" ")].rstrip(",;:") + "..."


_QUESTION_RE = re.compile(
    r"^\s*(who|what|whats|what's|when|where|why|how|which|did|does|do|is|are|was|were|can|"
    r"could|should|will)\b", re.I)


def _is_question(q: str) -> bool:
    """Question-shaped queries want an ANSWER; entity-shaped ones want a DEFINITION.
    Routing them to the wrong backend is how "who won the Oscar in 2019" became
    "Oscar Isaac is an American actor" -- see the note in _h_facts."""
    q = (q or "").strip()
    return bool(q.endswith("?") or _QUESTION_RE.match(q))


# "what is X" / "who is X" are DEFINITIONAL -- they name an entity and ask for a summary,
# which is precisely what Wikipedia's endpoint does well. Everything else that is
# question-shaped ("who won", "when did", "how many") asks for a SPECIFIC FACT, where a
# fuzzy entity match is a wrong answer rather than a partial one.
_DEFINITIONAL_RE = re.compile(
    r"^\s*(what|what's|whats|who)\s+(is|are|was|were)\s+(a|an|the)?\s*", re.I)


def _strip_definitional(q: str) -> str:
    """'what is a transformer neural network' -> 'transformer neural network'"""
    return _DEFINITIONAL_RE.sub("", (q or "").strip()).rstrip("?").strip() or q


def _h_wikidata(p):
    """Wikidata SPARQL -- structured facts Wikipedia's prose cannot answer.

    FREE, no key, no signup (query.wikidata.org). Answers exactly the class that defeated
    both other tools: "who won the Oscar for best actor in 2019" -> **Rami Malek**, because
    Wikidata stores it as a relation with a date qualifier rather than as prose.

    **MEASURED 8.6 SECONDS.** That is ~5x BMO's whole conversational tick (1.8 s), so this
    must NOT sit inline in a reply. It is registered as its own tool so the caller can decide
    to run it in the background and speak a `thinking_filler` backchannel meanwhile -- the
    mechanism `PrebuiltVoiceBank` already provides. Calling this on the critical path will
    make BMO appear frozen.

    Deliberately template-based, not general natural-language-to-SPARQL: that is an open
    research problem, and a wrong structured query returns a confidently wrong fact. Covers
    award-by-year, which is the pattern that motivated it; extend with more templates as
    real needs appear rather than speculatively."""
    q = (p.get("query") or p.get("q") or "").strip()
    if not q:
        return None
    m = re.search(r"(?:oscar|academy award).*?best\s+(actor|actress|director|picture)"
                  r".*?((?:19|20)\d{2})", q, re.I)
    if not m:
        return None
    role, year = m.group(1).lower(), int(m.group(2))
    QID = {"actor": "Q103916", "actress": "Q103618", "director": "Q103360",
           "picture": "Q102427"}[role]
    try:
        import json as _json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        # wdt:P31 wd:Q5 restricts to HUMANS -- without it the winning FILM comes back too
        # (measured: the 2019 query returned both "Rami Malek" and "Bohemian Rhapsody").
        human = "?who wdt:P31 wd:Q5 ." if role != "picture" else ""
        sq = (f'SELECT ?whoLabel WHERE {{ ?who p:P166 ?st . '
              f'?st ps:P166 wd:{QID} ; pq:P585 ?when . {human} '
              f'FILTER(YEAR(?when) = {year}) '
              f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} LIMIT 1')
        url = "https://query.wikidata.org/sparql?" + urlencode({"query": sq, "format": "json"})
        req = Request(url, headers={"User-Agent": "BMO-companion/1.0 (local robot assistant)",
                                    "Accept": "application/sparql-results+json"})
        with urlopen(req, timeout=15.0) as r:      # generous: the public endpoint is slow
            rows = _json.load(r)["results"]["bindings"]
        return rows[0]["whoLabel"]["value"] if rows else None
    except Exception:
        return None


def _h_lookup(p):
    """The tool the LLM should actually call. Routes, then refuses rather than guessing.

        question ("who won X in 2019")  -> DuckDuckGo instant answer, else None
        entity   ("Jetson Orin Nano")   -> Wikipedia summary, else DuckDuckGo, else None

    Returning None is a first-class outcome. Every fabricated answer this project has shipped
    -- the 72-degree weather stub, "Oscar Isaac" for an Oscars question -- came from a code
    path that preferred *something* over *nothing*."""
    q = (p.get("query") or p.get("q") or p.get("about") or p.get("topic") or "").strip()
    if not q:
        return None
    if _is_question(q):
        # DEFINITIONAL questions ("what is a transformer neural network") want an entity
        # summary, so they may fall back to Wikipedia. SPECIFIC-FACT questions ("who won X
        # in 2019") must NOT -- that fallback is exactly what produced "Oscar Isaac".
        if _DEFINITIONAL_RE.match(q):
            return _h_search(p) or _h_facts({"query": _strip_definitional(q)}) or None
        return _h_search(p) or None
    return _h_facts(p) or _h_search(p) or None


def _h_facts(p):
    """Wikipedia. Settles factual questions ("who won Best Actor in 2019").

    No API key, no signup -- which is exactly why it is the first retrieval tool worth
    adding. Two hops: search for the best-matching title, then pull that page's summary.
    Returns None on any failure so ToolDispatcher can say it cannot check, rather than
    inventing -- the same rule the weather handler now follows."""
    q = (p.get("query") or p.get("q") or p.get("about") or p.get("topic") or "").strip()
    if not q:
        return None
    try:
        import json as _json
        from urllib.parse import urlencode, quote
        from urllib.request import Request, urlopen
        hdrs = {"User-Agent": "BMO-companion/1.0 (local robot assistant)"}

        srch = ("https://en.wikipedia.org/w/api.php?"
                + urlencode({"action": "query", "list": "search", "srsearch": q,
                             "format": "json", "srlimit": 1}))
        with urlopen(Request(srch, headers=hdrs), timeout=4.0) as r:
            hits = _json.load(r)["query"]["search"]
        if not hits:
            return None
        title = hits[0]["title"]

        # RELEVANCE GUARD -- and a note on why the OBVIOUS guard does not work.
        # A lexical overlap check was tried first and is defeated by exactly the case that
        # motivated it: "who won the Oscar for best actor in 2019" returns the title
        # **"Oscar Isaac"**, which genuinely shares the token "oscar". Word overlap cannot
        # distinguish the award from the first name.
        #
        # The real error was architectural: that is a QUESTION, and this endpoint answers
        # ENTITY lookups. `_h_facts` is now only reached for entity-shaped queries (see
        # is_question below); question-shaped ones route to DuckDuckGo first and, if it has
        # no answer, BMO says it does not know instead of guessing at a page.
        _STOP = {"who", "what", "when", "where", "which", "the", "a", "an", "of", "in", "for",
                 "is", "was", "did", "does", "won", "win", "best", "and", "to", "on", "at"}
        qt = {w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in _STOP and len(w) > 2}
        tt = {w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 2}
        if qt and not (qt & tt):
            return None

        summ = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
                + quote(title.replace(" ", "_"), safe=""))
        with urlopen(Request(summ, headers=hdrs), timeout=4.0) as r:
            d = _json.load(r)
        extract = d.get("extract") or ""
        return _clip(extract) or None
    except Exception:
        return None


def _h_search(p):
    """DuckDuckGo Instant Answer API -- current events / open web.

    Brave's free tier is gone, so this is the no-key option. It is the *instant answer* box
    rather than a full web index, so it answers a subset of queries well and returns nothing
    for the rest -- which is fine here, because returning nothing is a supported outcome and
    fabricating is not. Falls back to a RelatedTopics blurb when there is no AbstractText."""
    q = (p.get("query") or p.get("q") or p.get("about") or "").strip()
    if not q:
        return None
    try:
        import json as _json
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        url = ("https://api.duckduckgo.com/?"
               + urlencode({"q": q, "format": "json", "no_html": 1,
                            "skip_disambig": 1, "t": "bmo-companion"}))
        with urlopen(Request(url, headers={"User-Agent": "BMO-companion/1.0"}),
                     timeout=4.0) as r:
            d = _json.load(r)
        txt = (d.get("AbstractText") or d.get("Answer") or "").strip()
        if not txt:
            for t in d.get("RelatedTopics", []):
                if isinstance(t, dict) and t.get("Text"):
                    txt = t["Text"]; break
        return _clip(txt) or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# POWER & COOLING (2026-08-16) -- BMO's own body.
#
# Every other tool here answers a question about the world. These are the first that let BMO
# act on ITSELF: read its battery, change its own power envelope, spin its own fan up or down.
# Backed by the real `bmo-power` CLI + `bmo_power` python module on the Jetson (INA219 on I2C
# bus 7 @ 0x41, Waveshare UPS Module C 3S pack, PWM fan hwmon0/pwm1 with tach feedback,
# nvpmodel for instant no-reboot mode switching).
#
# Import path first, subprocess second: the module is installed in site-packages on the Jetson
# and is cheaper, but the CLI is the contract that is guaranteed to exist and works from any
# interpreter. Both are wrapped so a missing tool degrades to None (no answer) rather than
# raising into the conversational tick -- same rule as _h_weather: never fabricate a reading.
# ---------------------------------------------------------------------------

_POWER_MODES = {"7w": "7W", "15w": "15W", "25w": "25W",
                "maxn": "MAXN_SUPER", "maxn_super": "MAXN_SUPER", "max": "MAXN_SUPER"}
_FAN_PROFILES = {"auto", "quiet", "cool", "max"}


def power_status() -> Optional[Dict]:
    """Full telemetry dict, or None if this is not the Jetson / tool is absent."""
    try:
        import bmo_power                                    # installed on-device
        return bmo_power.get_power_status()
    except Exception:
        pass
    try:
        import json as _json
        import subprocess
        out = subprocess.run(["bmo-power", "--json"], capture_output=True,
                             text=True, timeout=4.0)
        return _json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def battery_to_energy(status: Optional[Dict] = None) -> Optional[float]:
    """Map real battery state onto the homeostatic `energy` variable in [0, 1].

    ONE STATE DRIVES WORDS, VOICE AND FACE (the principle already used for mood), so BMO's
    felt energy should track its ACTUAL energy rather than a simulated decay. Charging is not
    clamped to 1.0 -- a robot on the charger at 30% is still low, it is just no longer
    falling -- but it gets a floor so BMO does not act exhausted while plugged in."""
    st = status if status is not None else power_status()
    if not st:
        return None
    b = st.get("battery") or {}
    if not b.get("connected"):
        return None
    pct = float(b.get("percentage", 0.0)) / 100.0
    charging = str(b.get("state", "")).upper() in ("CHARGING", "FULL")
    return max(pct, 0.45) if charging else pct


def _fmt_power(st: Dict, field: str) -> Optional[str]:
    b, pm, fan = st.get("battery") or {}, st.get("power_mode") or {}, st.get("fan") or {}
    f = (field or "all").lower()
    if f in ("battery", "percent", "percentage", "charge"):
        s = "charging" if str(b.get("state", "")).upper() in ("CHARGING", "FULL") else "on battery"
        return f"{b.get('percentage')}% and {s}"
    if f in ("voltage", "volts"):     return f"{b.get('voltage_v')} volts"
    if f in ("current", "amps"):      return f"{b.get('current_a')} amps"
    if f in ("power", "watts", "draw"): return f"{b.get('power_w')} watts"
    if f in ("mode", "power_mode"):   return f"{pm.get('mode_name')}"
    if f in ("fan", "fan_speed", "rpm"):
        return f"{fan.get('rpm')} rpm at {fan.get('pwm_percentage')}% ({fan.get('mode')})"
    if f in ("temp", "temperature", "temps", "thermal"):
        t = fan.get("temperatures") or {}
        return (f"CPU {t.get('cpu-thermal')}C, GPU {t.get('gpu-thermal')}C, "
                f"SOC {t.get('soc0-thermal')}C")
    if f in ("runtime", "remaining", "time_left"):
        return f"about {b.get('estimated_runtime_human')} left"
    return (f"{b.get('percentage')}% battery ({b.get('state_description')}), "
            f"{b.get('power_w')}W in {pm.get('mode_name')}, fan {fan.get('rpm')} rpm")


def _h_power(p):
    st = power_status()
    return _fmt_power(st, p.get("field") or p.get("what") or "all") if st else None


def _h_power_mode(p):
    """Switch nvpmodel power envelope (instant, no reboot), optionally with a fan profile."""
    raw = (p.get("mode") or p.get("to") or "").strip().lower().replace("-", "_")
    mode = _POWER_MODES.get(raw)
    if mode is None:
        return None                       # unknown mode -> say nothing rather than guess
    fan = (p.get("fan_profile") or p.get("fan") or "").strip().lower() or None
    if fan is not None and fan not in _FAN_PROFILES:
        fan = None
    try:
        import bmo_power
        bmo_power.set_power_mode(mode, fan_mode=fan) if fan else bmo_power.set_power_mode(mode)
        return f"power mode to {mode}" + (f" with the {fan} fan profile" if fan else "")
    except Exception:
        pass
    try:
        import subprocess
        cmd = ["bmo-power", "--set-mode", mode]
        if subprocess.run(cmd, capture_output=True, timeout=8.0).returncode != 0:
            return None
        if fan:
            subprocess.run(["bmo-power", "--fan", fan], capture_output=True, timeout=8.0)
        return f"power mode to {mode}" + (f" with the {fan} fan profile" if fan else "")
    except Exception:
        return None


def _h_fan(p):
    """Set fan speed or restore the closed-loop thermal governor.

    Also the lever the mic wants: the ReSpeaker sits ~10cm from the intake and the fan is the
    dominant sound in the room (measured -- see CLAUDE.md). Being able to say `quiet` before
    listening is a real perception fix, not just an acoustic nicety."""
    sp = (p.get("speed") or p.get("to") or p.get("profile") or "").strip().lower()
    if not sp:
        return None
    if sp not in _FAN_PROFILES and not re.fullmatch(r"\d{1,3}%?", sp):
        return None
    try:
        import bmo_power
        bmo_power.set_fan_speed(sp)
        return f"fan to {sp}"
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(["bmo-power", "--fan", sp], capture_output=True, timeout=8.0)
        return f"fan to {sp}" if r.returncode == 0 else None
    except Exception:
        return None


def power_guard(low_pct: float = 20.0, critical_pct: float = 10.0) -> Optional[Dict]:
    """The autonomous rules from the user's spec section 7, as data rather than prose.

    Returns None when there is nothing to do, so a caller can run this every tick cheaply and
    only react when it fires. Deliberately does NOT act on its own -- it reports the action and
    a line for BMO to say, and the caller decides. A companion silently throttling itself is
    indistinguishable from a companion getting slow and dull."""
    st = power_status()
    if not st:
        return None
    b = st.get("battery") or {}
    if not b.get("connected") or str(b.get("state", "")).upper() != "DISCHARGING":
        return None
    pct = float(b.get("percentage", 100.0))
    if pct <= critical_pct:
        return {"level": "critical", "battery_pct": pct, "mode": "7W", "fan": "quiet",
                "say": "My battery is really low -- can you plug me in?"}
    if pct <= low_pct:
        return {"level": "low", "battery_pct": pct, "mode": "7W", "fan": "quiet",
                "say": "I'm getting low on power, so I'm going into a quieter mode."}
    return None


class ToolRegistry:
    def __init__(self):
        self._h: Dict[str, Callable[[Dict[str, str]], str]] = {
            "time": _h_time, "date": _h_date, "weather": _h_weather,
            "timer": _h_timer, "reminder": _h_reminder, "search": _h_search,
            "facts": _h_lookup, "wiki": _h_facts, "lookup": _h_lookup, "wikidata": _h_wikidata,
            "power": _h_power, "battery": _h_power, "temperature": _h_power,
            "set_power_mode": _h_power_mode, "power_mode": _h_power_mode,
            "fan": _h_fan, "set_fan": _h_fan,
        }

    def register(self, name: str, handler: Callable[[Dict[str, str]], str]) -> None:
        self._h[name.lower()] = handler

    def has(self, name: str) -> bool:
        return name.lower() in self._h

    def execute(self, name: str, params: Dict[str, str]) -> Optional[str]:
        h = self._h.get(name.lower())
        if h is None:
            return None
        try:
            return h(params)
        except Exception as e:  # a tool failure must not crash the turn
            return None


# BMO-flavored little connectors so a templated result still sounds like BMO
_LEAD = {
    "time": "It's {r} right now!",
    "date": "Today is {r}.",
    "weather": "Beemo checked -- it's {r}!",
    "timer": "Okay! Beemo set {r}.",
    "reminder": "Got it! Beemo will remember {r}.",
    "search": "Beemo found {r}.",
    "power": "Beemo checked its own body -- {r}.",
    "battery": "Beemo's battery is {r}.",
    "temperature": "Beemo is running at {r}.",
    "set_power_mode": "Okay! Beemo switched its {r}.",
    "power_mode": "Okay! Beemo switched its {r}.",
    "fan": "Okay! Beemo turned its {r}.",
    "set_fan": "Okay! Beemo turned its {r}.",
}


class ToolDispatcher:
    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or ToolRegistry()

    def resolve(self, text: str, *, transcript: Optional[str] = None,
                generate_fn: Optional[Callable[[str, Dict], object]] = None,
                state: Optional[Dict] = None, use_llm_second_pass: bool = False):
        """Return (resolved_text, tool_events). If the LLM output has no tool
        calls, returns the text unchanged. Otherwise executes each tool and
        folds the result back in -- either by templating (default: reliable, no
        extra latency) or, if use_llm_second_pass and the model is trained to
        consume [tool_result ...] context, by a second fast-tier pass."""
        calls = parse_tool_calls(text)
        if not calls:
            return text, []

        events = []
        results = []
        for name, params in calls:
            res = self.registry.execute(name, params)
            events.append({"name": name, "params": params, "result": res})
            if res is not None:
                results.append((name, res))

        if use_llm_second_pass and generate_fn is not None and results:
            ctx = " ".join(f"[tool_result {n}: {r}]" for n, r in results)
            aug = f"{transcript or strip_tool_calls(text)}\n{ctx}"
            r2 = generate_fn(aug, state or {})
            return getattr(r2, "text", str(r2)), events

        # template mode: keep any pre-call BMO chatter, append the folded result
        lead_text = strip_tool_calls(text)
        pieces = [lead_text] if lead_text else []
        for name, res in results:
            pieces.append(_LEAD.get(name, "{r}").format(r=res))
        resolved = " ".join(p for p in pieces if p).strip()
        return (resolved or lead_text or text), events
