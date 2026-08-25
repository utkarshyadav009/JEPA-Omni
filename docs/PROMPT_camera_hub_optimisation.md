# Task prompt — fix and optimise the BMO camera pipeline

Copy everything below to the agent doing this work.

---

## Your job

On the Jetson Orin Nano (`ssh bmo@bmo-desktop`, Tailscale), the camera pipeline has **one
blocking defect** and **~476 MiB of recoverable memory**. Fix the defect first, then optimise.

Do not re-derive the numbers below — they were measured on this device on 2026-08-24 and the
commands are given so you can re-verify rather than re-discover.

---

## 1. THE BLOCKING DEFECT — the camera dies silently under memory pressure

When the full BMO stack loads (`python3 ~/bmo_showcase.py --perception --identity`), the camera
hub loses its stream and **does not crash**:

```
NvMapMemHandleAlloc: error 0
NvRmStream: Buffer allocation failed (err=6)
(Argus) Error InsufficientMemory:  (propagating from src/eglstream/ImageImpl.cpp, line 482)
Error generated ... gstnvarguscamerasrc.cpp, threadExecute:691 IImageNativeBuffer not supported by Image
```

Observed behaviour: `/dev/shm/bmo_motion.txt` froze at a fixed timestamp and never updated
again, while `bmo_camera_hub.py` stayed alive at **8.5% CPU** with a dead pipeline. Eye tracking
stopped; anything reading `/tmp/bmo_cam_perception.sock` got nothing. It recovers only on a
manual restart, and only once memory is free (verified: restarted at 5,181 MiB free → motion
timestamps current, zero errors).

**Reproduce:**
```bash
nohup setsid python3 ~/bmo_production/scripts/bmo_camera_hub.py > ~/hub.log 2>&1 &
sleep 12; stat -c %y /dev/shm/bmo_motion.txt          # should tick at ~30 Hz
python3 ~/bmo_showcase.py --selftest --perception --identity   # loads the full stack
stat -c %y /dev/shm/bmo_motion.txt; grep -i insufficient ~/hub.log
```

### Required outcome
1. **The camera must survive full-stack load.** Constrain what Argus allocates — see §3.
2. **If it ever does die, it must die LOUDLY.** Add a watchdog: if `bmo_motion.txt` has not been
   written for N seconds while the pipeline claims to be PLAYING, log the reason and `exit(1)`
   so systemd restarts it. A process running dead at 8.5% CPU is the worst possible failure
   mode — silent, and invisible to every health check.
3. **Handle the GStreamer bus.** The hub currently connects only `new-sample` and never watches
   the bus, so `GST_MESSAGE_ERROR`/`EOS` pass unnoticed. That is why the failure was silent.
   Add a bus watch that logs and exits non-zero.

---

## 2. MEASURED MEMORY — verify before trusting any earlier figure

An earlier report gave hub 88.9 MB / nvargus 285 MB. Directly measured on 2026-08-24:

| process | RSS | how |
|---|---:|---|
| `bmo_camera_hub.py` | **142 MiB** | `grep VmRSS /proc/$(pgrep -f bmo_camera_hub.py \| tail -1)/status` |
| `nvargus-daemon` | **334 MiB** streaming (133 MiB idle) | same, on `pgrep -x nvargus-daemon` |
| **total** | **~476 MiB** | |

Beware: `pgrep -f bmo_camera_hub.py` matches the `setsid`/shell wrapper too. Take the **last**
PID or match on the python interpreter, otherwise you will measure a 32 kB wrapper and conclude
the hub is free. This already happened once.

The full stack leaves **279 MiB** free with the camera running, so every MiB recovered here is
directly useful.

---

## 3. OPTIMISATIONS, IN THE ORDER THEY ARE WORTH DOING

### 3.1 Constrain Argus allocation — this is the FIX, not a nice-to-have
`nvarguscamerasrc` defaults to querying the sensor's full mode and allocating a large pool of
NVMM DMA buffers. The hub only ever needs 1280×720.

* Pin the sensor mode explicitly (`sensor-mode=<n>` — enumerate with
  `gst-inspect-1.0 nvarguscamerasrc` and the boot log's `GST_ARGUS: <W> x <H> FR = ..` lines;
  a 1280×720 mode was reported available at 59.99 fps).
* Reduce buffer counts: `nvarguscamerasrc bufapi-version`, and on every `queue`
  `max-size-buffers=1 leaky=downstream` (the hub already does this on its two branches).
* Consider `wbmode`/`aelock` only if it reduces ISP working set — measure, do not assume.

**Verify NVMM specifically, not just RSS:** `cat /sys/kernel/debug/nvmap/iovmm/clients`
(needs root) shows per-client NVMM. RSS will not show the carveout.

### 3.2 Rewrite the hub in C++ — worth ~127 MiB
The hub is **142 MiB**, almost all Python interpreter + PyGObject + NumPy. An equivalent
GStreamer C API binary lands around 12–15 MiB. This is a bigger win than the ~75 MiB previously
estimated, because the starting figure was wrong.

Keep the behaviour identical:
* single `nvarguscamerasrc` → `tee`
* branch A: 160×120 GRAY8 → frame-difference motion centroid → `/dev/shm/bmo_motion.txt`
  as `"<0|1> <cx> <cy>\n"`, cx/cy in [-1,1], X flipped (camera faces the user)
* branch B: 256×256 BGRx → `shmsink socket-path=/tmp/bmo_cam_perception.sock`
* `flip-method=2` (the CSI module is mounted upside down — this is settled, do not change it)

Use ARM NEON for the frame difference if convenient, but the current motion cost is not the
bottleneck; memory is.

### 3.3 Add the identity face crop to the hub
`m5_motion_crop.py` computes a 224×224 crop around the motion centroid with `HEAD_BIAS=0.22`.
Doing it in the hub (a third `tee` branch or derived from branch A's centroid) means the
perception process never needs a second full-resolution frame.

### 3.4 Install the systemd unit — it was never installed
`bmo_camera_hub.service` exists in the repo but **is not in `/etc/systemd/system`**, so
`systemctl start bmo_camera_hub` fails with "Unit not found" and the hub only ever runs when
started by hand. Install it, with:
* `Restart=always`, `RestartSec=2`
* `StartLimitBurst=5` / `StartLimitIntervalSec=60` — so a genuinely broken camera cannot
  crash-loop a core. (The face engine did exactly this **16,489 times** before it was noticed.)
* ordering after `nvargus-daemon.service`

### 3.5 Only then consider a POSIX shm ring instead of `shmsink`
A 256×256×3 frame is 192 KB. A fixed shared buffer with an atomic frame counter removes the
GStreamer socket IPC. Worth ~2–4 MiB and some latency. **Low priority** — do not spend time here
until 3.1–3.4 are done.

---

## 4. DO NOT DO THESE

* **Do not fold capture into the perception process** ("in-process capture"). The CSI sensor is
  **exclusive** — one process may open it. The face engine's eye tracking needs
  `/dev/shm/bmo_motion.txt`, which `motion_tracker` produces by owning the camera. The hub
  exists precisely so eye tracking and perception can coexist, and that coexistence is what
  made the full stack fit at all. Folding capture back in re-breaks it, and the process still
  has to do the work, so the saving is illusory.
* **Do not bypass Argus with raw V4L2** to "save 334 MiB". `nvargus-daemon` is a system daemon
  that keeps running regardless, so the saving is far smaller than it looks, and you lose
  hardware auto-exposure and white-balance. The room is dim; auto-exposure matters.
* **Do not change `flip-method=2`.** The module is physically mounted upside down. A previous
  wrong orientation produced a confident, wrong perception answer (`a person lying down +0.71`
  for a seated person).
* **Do not `pkill -f` anything over SSH.** It matches your own session's argv and kills your
  shell. This has happened **seven** times in this project. Signal PIDs, and wait on **log
  markers**, never process names.

---

## 5. DEFINITION OF DONE

All of these must hold, demonstrated with commands and output pasted into your report:

1. `bmo_motion.txt` timestamps keep advancing **throughout** a full
   `bmo_showcase.py --perception --identity` boot **and** a subsequent live turn — not just at
   idle. Zero `InsufficientMemory` in the hub log.
2. A deliberately induced camera failure (e.g. stop `nvargus-daemon`) causes the hub to **exit
   non-zero within N seconds** and systemd to restart it. Show the journal.
3. `bmo_camera_hub` RSS reported before and after, measured on the **python/binary PID**, not a
   wrapper.
4. NVMM usage before and after from `/sys/kernel/debug/nvmap/iovmm/clients`.
5. A perception client still reads real frames from the socket: non-zero variance, correct
   256×256×3 shape, ~30 fps. A black or frozen frame passes a naive "did it return an array"
   check — assert on **content and freshness**, not just success.
6. Free memory at full-stack boot, before vs after your changes. Baseline to beat: **279 MiB**.

**A green test that cannot see the main failure mode is worse than no test.** The existing
`--selftest` reported PASS while the camera was dead, because it never pulls a frame. If you add
tests, make them fail when the camera is dead.
