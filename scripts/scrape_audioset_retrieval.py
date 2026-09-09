"""Scrape audio+video for the balanced AudioSet eval retrieval gallery (Part C).

Downloads only the labelled 10 s segment of each clip via yt-dlp --download-sections,
so bandwidth is spent on the window that carries the label rather than whole videos.
Writes one mp4 per clip and appends a JSONL progress record per attempt, so per-class
survival can be computed from a partial run -- classes are never padded to hit a target.
"""
import json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

OUT = "/mnt/Raid-Storage-2/utkarsh-data/audioset_scrape"
LOG = "/mnt/Raid-Storage-2/utkarsh-data/audioset_scrape/progress.jsonl"
TARGET = "/home/utkarsh/JEPA-Omni/data/audioset_retrieval_target.json"
WORKERS = 8
YTDLP = "/home/utkarsh/miniconda3/envs/jepa-omni/bin/yt-dlp"
lock = threading.Lock()
done = {"ok": 0, "fail": 0, "t0": time.time()}


def fetch(rec):
    ytid, st = rec["ytid"], rec["start"]
    dst = os.path.join(OUT, f"{ytid}.mp4")
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        return ("cached", ytid, "")
    # yt-dlp's --download-sections ranged fetch returns "ffmpeg exited with code 8", and the
    # default client returns HTTP 403 on video data (metadata fetch still succeeds). The
    # android player client downloads fine, so: fetch the whole low-res video, then trim the
    # labelled 10 s window locally with ffmpeg.
    tmp = dst + ".full.mp4"
    cmd = [YTDLP, "--quiet", "--no-warnings", "--no-playlist",
           "--extractor-args", "youtube:player_client=android",
           "-f", "b[height<=360]/b",
           "-o", tmp, f"https://www.youtube.com/watch?v={ytid}"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=240)
        ok = r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 10000
        err = "" if ok else (r.stderr.decode()[-160:] if r.stderr else "no file")
        if ok:
            t = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{st:.1f}",
                                "-t", "10", "-i", tmp, "-c", "copy", dst],
                               capture_output=True, timeout=120)
            if not (os.path.exists(dst) and os.path.getsize(dst) > 10000):
                t = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{st:.1f}",
                                    "-t", "10", "-i", tmp, dst], capture_output=True, timeout=180)
            ok = os.path.exists(dst) and os.path.getsize(dst) > 10000
            if not ok:
                err = "trim failed: " + (t.stderr.decode()[-120:] if t.stderr else "")
        try:
            os.remove(tmp)
        except OSError:
            pass
    except subprocess.TimeoutExpired:
        ok, err = False, "timeout"
    except Exception as e:
        ok, err = False, repr(e)[:160]
    with lock:
        done["ok" if ok else "fail"] += 1
        n = done["ok"] + done["fail"]
        el = (time.time() - done["t0"]) / 3600.0
        with open(LOG, "a") as f:
            f.write(json.dumps({"ytid": ytid, "ok": ok, "err": err,
                                "labels": rec["labels"]}) + "\n")
        if n % 25 == 0:
            print("[scrape] %d/%d  ok=%d fail=%d  %.0f clips/hr  elapsed %.2f h"
                  % (n, TOTAL, done["ok"], done["fail"], n / max(el, 1e-6), el), flush=True)
    return ("ok" if ok else "fail", ytid, err)


if __name__ == "__main__":
    recs = json.load(open(TARGET))
    TOTAL = len(recs)
    print("[scrape] target %d clips, %d workers -> %s" % (TOTAL, WORKERS, OUT), flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(fetch, recs))
    el = (time.time() - done["t0"]) / 3600.0
    print("[scrape] FINISHED ok=%d fail=%d of %d in %.2f h (%.0f/hr)"
          % (done["ok"], done["fail"], TOTAL, el, TOTAL / max(el, 1e-6)), flush=True)
