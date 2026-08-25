"""models/m5_identity_schedule.py — run identity when the answer can have CHANGED, not every tick.

MEASURED PROBLEM. With face cropping enabled, an identity query costs **218-239 ms** on the
Jetson, because the crop has to go through a second V-JEPA2 forward. Without cropping it is
4-9 ms, but then the head sees a whole 1280x720 frame instead of the localised crop it was
trained closer to. So the good path is ~30x the price of the cheap one, and it was being paid
on every tick (measured: total tick 1,891 ms -> 2,295 ms).

THE INSIGHT IS BORROWED FROM REAL-TIME RENDERING, not from ML. "Who am I talking to" changes
on the scale of minutes; the tick runs at ~2 s. Recomputing it every frame is exactly the
redundancy that temporal reuse / amortized shading exists to remove: do the expensive work
when the input actually changed, reproject (here: reuse) otherwise, and gate it behind a
cheap always-on signal. The cheap signal already exists and is free -- the motion centroid
from models/m5_motion_crop.py, which the capture thread computes anyway.

WHEN THIS SAYS YES:
  1. **Nothing known yet** -- there is no cached answer to reuse.
  2. **Motion onset** -- the scene went from still to moving. Someone arrived, turned, or
     leaned in. This is the case that matters: it is when a NEW person is most likely, and
     it is also when a crop is actually available (a still person produces no centroid).
  3. **Staleness** -- `max_age_s` has passed. Guards against a silent swap during continuous
     motion, where onset never re-fires.
  4. **Low confidence** -- the cached answer was weak, so retry sooner (`retry_unknown_s`)
     rather than holding "I don't know you" for the full refresh period.

Everything else reuses the cached answer at zero cost.

DELIBERATELY NOT DONE HERE: no averaging or voting across ticks. Enrolment quality is a
separate, already-measured axis (1 clip 0.584 -> 8 clips 0.760 TAR@FAR1%); mixing that into
the schedule would conflate "when to look" with "how sure to be", and the memory's own
three-way output (match / below_threshold / ambiguous) is what should carry uncertainty.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IdentityDecision:
    who: Optional[str]
    confidence: float
    reason: str            # from JepaMemory.query: match / below_threshold / ambiguous / empty_memory
    age_s: float           # how old this answer is
    recomputed: bool       # True if the expensive path ran on this tick


@dataclass
class IdentitySchedule:
    """Gate for the expensive identity path. `should_run()` is pure and cheap; the caller
    owns the model work and hands the result back via `commit()`."""

    max_age_s: float = 30.0        # force a refresh even under continuous motion
    retry_unknown_s: float = 8.0   # a weak/unknown answer is retried sooner than a confident one
    min_confidence: float = 0.5    # below this an answer counts as "not really known"
    require_motion: bool = True    # only crop when a centroid exists (a still person has none)

    _who: Optional[str] = None
    _conf: float = 0.0
    _reason: str = "never_run"
    _stamp: float = 0.0
    _was_moving: bool = False
    n_run: int = 0
    n_skipped: int = 0

    @property
    def age_s(self) -> float:
        return float("inf") if self._stamp == 0.0 else time.time() - self._stamp

    def should_run(self, moving: bool, now: Optional[float] = None) -> tuple[bool, str]:
        """-> (run?, why). `moving` is MotionCentroid.last.moving."""
        now = now or time.time()
        onset = moving and not self._was_moving      # evaluate BEFORE updating the edge state
        self._was_moving = moving

        if self._stamp == 0.0:
            return True, "no_answer_yet"
        age = now - self._stamp
        if onset and (not self.require_motion or moving):
            return True, "motion_onset"
        if self._conf < self.min_confidence and age >= self.retry_unknown_s:
            return True, "low_confidence_retry"
        if age >= self.max_age_s:
            return True, "stale"
        return False, "reuse_cached"

    def commit(self, who: Optional[str], confidence: float, reason: str) -> IdentityDecision:
        """Record the result of an expensive run."""
        self._who, self._conf, self._reason = who, float(confidence), reason
        self._stamp = time.time()
        self.n_run += 1
        return IdentityDecision(who, self._conf, reason, 0.0, True)

    def cached(self) -> IdentityDecision:
        """Reuse the previous answer. Costs nothing."""
        self.n_skipped += 1
        return IdentityDecision(self._who, self._conf, self._reason, self.age_s, False)

    def stats(self) -> dict:
        tot = self.n_run + self.n_skipped
        return {"runs": self.n_run, "skipped": self.n_skipped,
                "run_rate": (self.n_run / tot) if tot else 0.0}


if __name__ == "__main__":
    s = IdentitySchedule(max_age_s=5.0, retry_unknown_s=2.0)

    print("cold start          ->", s.should_run(moving=False))
    s.commit("Alice", 0.9, "match")
    print("still, fresh        ->", s.should_run(moving=False))
    print("still again         ->", s.should_run(moving=False))
    print("MOTION ONSET        ->", s.should_run(moving=True))
    s.commit("Alice", 0.9, "match")
    print("still moving (no onset) ->", s.should_run(moving=True))

    s2 = IdentitySchedule(retry_unknown_s=0.0)
    s2.commit(None, 0.0, "below_threshold")
    print("weak answer retries ->", s2.should_run(moving=False))

    s3 = IdentitySchedule(max_age_s=0.0)
    s3.commit("Bob", 0.99, "match")
    print("stale forces rerun  ->", s3.should_run(moving=False))
    print("stats:", s.stats())
