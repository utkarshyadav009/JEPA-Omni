"""models/m5_live_capture.py — item 6b: real live camera + microphone
capture, feeding models.m5_streaming_loop.StreamingLoop's rolling buffers
at real hardware cadence instead of the dummy-frame/simulated-audio
harnesses every prior Jetson script in this project has used.

STATUS: UNTESTED end-to-end. This Jetson currently has NO camera attached
(`/dev/video*` does not exist; `lsusb` shows no webcam) and no confirmed
physical microphone (`arecord -l` only lists the Tegra APE's internal
audio-routing channels, not a USB/analog capture device). Written now per
explicit instruction ("write capture code now, wait on hardware") so it is
ready the moment a USB webcam + mic are attached -- every number this
module could produce (real frame rate achieved, real capture latency) is
NOT RUN until then. Do not report any such number as measured.

Two classes, each a background thread pushing real captured data into an
existing StreamingLoop instance's buffers (ingest_video_frame /
ingest_audio_chunk) -- no change to StreamingLoop itself, this is a
capture-side adapter only.

  LiveCameraCapture: OpenCV (cv2.VideoCapture, already installed on this
    Jetson, v4.8.0) against a USB/V4L2 device index. Resizes/center-crops
    to the (3,256,256) uint8 format VisionEncoder.encode() expects (same
    convention as data/video_text_dataset.py's frame preprocessing).
    NOTE: this targets a V4L2 (USB) webcam via cv2's default backend. A
    CSI camera (ribbon-cable, e.g. IMX219) would need a GStreamer
    (nvarguscamerasrc) pipeline instead -- not written here since we don't
    yet know which kind of camera will be attached.

  LiveMicCapture: sounddevice (PortAudio; installed this session,
    pip3 install --user sounddevice) reads real 16kHz mono int16 chunks
    from the system's default input device, converts to the float32
    RollingAudioBuffer expects, and pushes into BOTH of StreamingLoop's
    audio paths via ingest_audio_chunk() (which already fans out to both
    the 2s Whisper buffer and the 10s ambient buffer internally -- see
    models/m5_streaming_loop.py's ingest_audio_chunk).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import torch


class LiveCameraCapture:
    """Reads frames from a real camera device at its native rate, pushing
    each into stream.ingest_video_frame(). Runs on its own thread so
    capture cadence is decoupled from tick()/vision-refresh cadence,
    matching the project's existing decoupled-perception-refresh design.

    device_index: cv2.VideoCapture index (0 = first V4L2 device, i.e.
    /dev/video0). target_fps: throttle capture to this rate even if the
    camera can go faster (StreamingConfig.video_fps=6.4 is what the
    rolling buffer actually consumes; capturing much faster than that
    just wastes CPU on frames that get overwritten in RollingVideoBuffer's
    fixed-length deque before ever being sampled)."""

    def __init__(self, stream, device_index: int = 4, target_fps: float = 6.4,
                 resolution: int = 256):
        # device_index default (2026-08-01, real hardware confirmed): the attached
        # camera is an Intel RealSense D435i, which exposes SIX /dev/video* nodes,
        # not one. Index 4 is the 1920x1080 RGB color stream (confirmed by direct
        # capture + auto-exposure ramp-up over the first ~10 frames). Indices
        # 0/1/3/5 report isOpened()=True but EVERY cap.read() call fails (ok=False)
        # -- they need a specific pixel-format request this class doesn't make, and
        # silently produce ZERO frames forever if selected, since _run only pushes
        # on ok=True and never raises for a persistently-failing read. Index 2 is a
        # single-channel 1280x720 stream (infrared), not RGB. If a different
        # RealSense unit or a plain UVC webcam is attached instead, re-run the
        # per-index probe (grab 15+ frames per index, check shape/channel-count/
        # rising mean brightness) before trusting index 4 again.
        self.stream = stream
        self.device_index = device_index
        self.target_fps = target_fps
        self.resolution = resolution
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.n_frames_captured = 0
        self.last_capture_t: Optional[float] = None

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        # center-crop to square, then resize -- same convention as
        # data/video_text_dataset.py's frame preprocessing
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        cropped = frame_rgb[top:top + side, left:left + side]
        resized = cv2.resize(cropped, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)
        chw = torch.from_numpy(resized).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        return chw

    def start(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera device index {self.device_index} -- "
                f"check /dev/video* exists and is accessible (v4l2-ctl --list-devices)")
        # isOpened()=True is NOT sufficient on a multi-stream device (e.g. RealSense
        # D435i's non-RGB sub-streams all open() fine but every read() fails) --
        # confirm at least one real frame comes back before starting the background
        # thread, so a wrong index fails loudly here instead of silently producing
        # zero frames forever.
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"camera device index {self.device_index} opened but read() failed -- "
                f"wrong sub-stream index for a multi-stream device? probe other indices "
                f"(grab 15+ frames, check shape/rising brightness) before retrying")

        def _loop():
            period = 1.0 / self.target_fps
            while not self._stop.is_set():
                t0 = time.perf_counter()
                ok, frame = cap.read()
                if ok:
                    chw = self._preprocess(frame)
                    self.stream.ingest_video_frame(chw)
                    self.n_frames_captured += 1
                    self.last_capture_t = time.time()
                elapsed = time.perf_counter() - t0
                self._stop.wait(max(0.0, period - elapsed))
            cap.release()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class LiveMicCapture:
    """Reads real audio from the ReSpeaker 4 Mic Array via sounddevice
    (PortAudio), pushes into stream.ingest_audio_chunk() at
    StreamingConfig.tick_interval_sec cadence (0.25s chunks by default) --
    ingest_audio_chunk already fans this out to BOTH the 2s Whisper/
    decision buffer and the 10s WavJEPA/ambient buffer internally."""

    def __init__(self, stream, sample_rate: int = 16000, chunk_sec: float = 0.25,
                 device: Optional[int] = 24):
        # device default (2026-08-01, real hardware confirmed): sounddevice index 24
        # is "ReSpeaker 4 Mic Array (UAC1.0)" (in=6, out=2 -- it's also the playback
        # device TTSEngine should target, see models/m5_tts.py). Requires
        # libportaudio2 installed system-side (not just the `sounddevice` pip
        # package) -- PortAudio raised OSError('PortAudio library not found') until
        # this was installed via apt on the Jetson. Do not rely on device=None
        # (system default) resolving to this device; verify with
        # sounddevice.query_devices() if the attached hardware changes.
        self.stream = stream
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.device = device
        self._stream_obj = None
        self.n_chunks_captured = 0
        self.last_capture_t: Optional[float] = None

    def start(self) -> None:
        import sounddevice as sd
        blocksize = int(self.sample_rate * self.chunk_sec)

        def _callback(indata, frames, time_info, status):
            if status:
                print(f"[live-mic] sounddevice status: {status}", flush=True)
            mono = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)
            self.stream.ingest_audio_chunk(mono.copy())
            self.n_chunks_captured += 1
            self.last_capture_t = time.time()

        self._stream_obj = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=blocksize, device=self.device, callback=_callback,
        )
        self._stream_obj.start()

    def stop(self) -> None:
        if self._stream_obj is not None:
            self._stream_obj.stop()
            self._stream_obj.close()


def list_available_devices() -> dict:
    """Diagnostic helper: what cameras/mics does THIS machine actually
    see right now? Run this before start()-ing either capture class on a
    new machine -- do not assume device_index=0 / default input exists."""
    result = {"cameras": [], "audio_input_devices": []}
    try:
        import cv2
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                result["cameras"].append(i)
            cap.release()
    except Exception as e:
        result["camera_check_error"] = repr(e)
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                result["audio_input_devices"].append({"index": i, "name": d["name"],
                                                        "max_input_channels": d["max_input_channels"]})
    except Exception as e:
        result["audio_check_error"] = repr(e)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(list_available_devices(), indent=2))
