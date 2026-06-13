"""MSR-VTT video/text dataset.

A :class:`torch.utils.data.Dataset` that yields ``(frames, caption)`` pairs
where ``frames`` is a ``uint8`` tensor of shape ``[T, C, H, W]`` with
``T = num_frames`` frames sampled uniformly across the clip and resized so the
SigLIP2 processor receives ``resolution x resolution`` inputs.

Decoding is done with :class:`torchcodec.decoders.VideoDecoder`.

The accompanying :func:`collate_fn` returns ``(list_of_frame_tensors,
list_of_captions)`` -- it deliberately does **not** stack the frames into a
single batch tensor, because :meth:`VisionEncoder.encode` consumes a *list* of
per-clip tensors (clips may differ in length / resolution).

Run this file directly for a quick sanity check on a tiny subset::

    python -m data.video_text_dataset \
        --video-dir /data/msrvtt/videos \
        --annotation /data/msrvtt/test_videodatainfo.json \
        --limit 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".webm", ".mkv", ".avi", ".mov")

# Common column name candidates for CSV-style annotations (e.g. JSFUSION test).
_CSV_ID_KEYS = ("video_id", "videoid", "vid_key", "key", "id")
_CSV_TEXT_KEYS = ("sentence", "caption", "text", "description")


@dataclass
class Sample:
    """A single (video, caption) training/eval example."""

    video_path: str
    caption: str
    video_id: str


def _uniform_frame_indices(num_total: int, num_frames: int) -> List[int]:
    """Return ``num_frames`` integer indices spread uniformly over the clip.

    When the clip has fewer frames than requested, indices repeat (so the
    output always has exactly ``num_frames`` entries).
    """
    if num_total <= 0:
        raise ValueError(f"Clip has no frames (num_total={num_total}).")
    if num_total == 1:
        return [0] * num_frames
    idx = torch.linspace(0, num_total - 1, steps=num_frames)
    idx = idx.round().long().clamp_(0, num_total - 1)
    return idx.tolist()


def _resolve_video_path(video_dir: str, video_id: str) -> str:
    """Resolve a ``video_id`` to a file path, trying common extensions."""
    # If the id already carries a known extension, trust it.
    if os.path.splitext(video_id)[1].lower() in VIDEO_EXTENSIONS:
        return os.path.join(video_dir, video_id)
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(video_dir, video_id + ext)
        if os.path.exists(candidate):
            return candidate
    # Fall back to .mp4 (decode will surface a clear error if it is missing).
    return os.path.join(video_dir, video_id + ".mp4")


def _parse_videodatainfo_blob(
    blob: Any,
    split_filter: Optional[str],
) -> List[Tuple[str, str]]:
    """Parse a MSR-VTT videodatainfo dict or list into (video_id, caption)."""
    if isinstance(blob, list):
        pairs: List[Tuple[str, str]] = []
        for item in blob:
            vid = item.get("video_id") or item.get("video")
            captions = item.get("caption", [])
            
            if isinstance(captions, str):
                captions = [captions]
                
            if vid and captions:
                clean_vid = os.path.splitext(str(vid))[0]
                for cap in captions:
                    if str(cap).strip():
                        pairs.append((clean_vid, str(cap).strip()))
        return pairs
    videos = blob.get("videos", [])
    sentences = blob.get("sentences", [])

    keep_ids = None
    if split_filter is not None and videos:
        keep_ids = {
            v["video_id"]
            for v in videos
            if str(v.get("split", "")).lower() == str(split_filter).lower()
        }

    pairs: List[Tuple[str, str]] = []
    for sent in sentences:
        vid = sent["video_id"]
        if keep_ids is not None and vid not in keep_ids:
            continue
        pairs.append((vid, sent["caption"].strip()))
    return pairs


def _parse_videodatainfo(
    annotation_file: str,
    video_dir: str,
    split_filter: Optional[str],
) -> List[Tuple[str, str]]:
    """Parse a MSR-VTT ``*_videodatainfo.json`` file into (video_id, caption)."""
    with open(annotation_file, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return _parse_videodatainfo_blob(blob, split_filter)


def _parse_vatex_blob(blob: Any) -> List[Tuple[str, str]]:
    """Parse VATEX JSON blob into (video_id, caption) pairs."""
    pairs: List[Tuple[str, str]] = []
    if isinstance(blob, list):
        for item in blob:
            vid = item.get("videoID")
            encap = item.get("enCap", [])
            if isinstance(encap, str):
                encap = [encap]
            if vid and encap:
                clean_vid = os.path.splitext(str(vid))[0]
                for cap in encap:
                    if str(cap).strip():
                        pairs.append((clean_vid, str(cap).strip()))
    return pairs


def _parse_vatex(annotation_file: str) -> List[Tuple[str, str]]:
    """Parse VATEX JSON file into (video_id, caption) pairs."""
    with open(annotation_file, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return _parse_vatex_blob(blob)


def _detect_and_parse_json(
    annotation_file: str,
    video_dir: str,
    split_filter: Optional[str],
) -> List[Tuple[str, str]]:
    """Detect JSON format (MSR-VTT or VATEX) and parse it."""
    lower_name = os.path.basename(annotation_file).lower()
    if "vatex" in lower_name:
        return _parse_vatex(annotation_file)
    if "msrvtt" in lower_name or "videodatainfo" in lower_name:
        return _parse_videodatainfo(annotation_file, video_dir, split_filter)

    # Fallback to structure-based detection
    with open(annotation_file, "r", encoding="utf-8") as fh:
        blob = json.load(fh)

    if isinstance(blob, dict):
        if "videos" in blob or "sentences" in blob:
            return _parse_videodatainfo_blob(blob, split_filter)
    elif isinstance(blob, list):
        if not blob:
            return []
        first = blob[0]
        if isinstance(first, dict):
            if "videoID" in first or "enCap" in first:
                return _parse_vatex_blob(blob)
            else:
                return _parse_videodatainfo_blob(blob, split_filter)

    return _parse_videodatainfo_blob(blob, split_filter)



def _parse_csv(annotation_file: str) -> List[Tuple[str, str]]:
    """Parse a CSV annotation (e.g. JSFUSION test split) into (video_id, caption)."""
    pairs: List[Tuple[str, str]] = []
    with open(annotation_file, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [f.lower() for f in (reader.fieldnames or [])]
        id_key = next((k for k in _CSV_ID_KEYS if k in fieldnames), None)
        text_key = next((k for k in _CSV_TEXT_KEYS if k in fieldnames), None)
        if id_key is None or text_key is None:
            raise ValueError(
                f"Could not find id/caption columns in {annotation_file!r}. "
                f"Columns present: {reader.fieldnames}"
            )
        # Map back to the original-cased field names.
        original = {f.lower(): f for f in (reader.fieldnames or [])}
        for row in reader:
            vid = str(row[original[id_key]]).strip()
            cap = str(row[original[text_key]]).strip()
            if vid and cap:
                pairs.append((vid, cap))
    return pairs


class MSRVTTVideoTextDataset(Dataset):
    """MSR-VTT (video, caption) dataset returning uint8 frame tensors.

    Parameters
    ----------
    video_dir:
        Directory containing the video files (``{video_id}.mp4`` etc.).
    annotation_file:
        Path to a MSR-VTT ``*_videodatainfo.json`` file or a CSV with
        ``video_id`` / ``sentence`` columns.
    num_frames:
        ``T`` -- number of frames sampled uniformly across each clip.
    resolution:
        Output spatial size; frames are resized to ``resolution x resolution``.
    split_filter:
        Keep only videos whose ``split`` field equals this value (JSON only).
        ``None`` keeps everything.
    limit:
        If set, truncate to the first ``limit`` samples (tiny debug subset).
    captions_per_video:
        If set (e.g. ``1`` for evaluation), keep at most this many captions per
        video, giving a clean 1:1 video<->text correspondence for retrieval.
    decode_device:
        Device passed to ``VideoDecoder`` ("cpu" or e.g. "cuda").
    require_exists:
        If ``True``, drop samples whose video file does not exist on disk.
    seed:
        RNG seed used for fallback re-sampling on decode errors.
    """

    def __init__(
        self,
        video_dir: str | List[str],
        annotation_file: str | List[str],
        num_frames: int,
        resolution: int,
        *,
        split_filter: Optional[str] = None,
        limit: Optional[int] = None,
        captions_per_video: Optional[int] = None,
        decode_device: str = "cpu",
        require_exists: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if resolution <= 0:
            raise ValueError("resolution must be positive.")

        self.video_dir = video_dir
        self.annotation_file = annotation_file
        self.num_frames = int(num_frames)
        self.resolution = int(resolution)
        self.decode_device = decode_device
        self._rng = random.Random(seed)

        # Handle list vs string for multi-dataset ingestion
        if isinstance(annotation_file, list):
            annos = annotation_file
        else:
            annos = [annotation_file]

        if isinstance(video_dir, list):
            vdirs = video_dir
        else:
            vdirs = [video_dir]

        if len(vdirs) == 1 and len(annos) > 1:
            vdirs = vdirs * len(annos)
        elif len(annos) == 1 and len(vdirs) > 1:
            annos = annos * len(vdirs)

        if len(vdirs) != len(annos):
            raise ValueError(
                f"Mismatch between number of video_dirs ({len(vdirs)}) and annotation_files ({len(annos)})"
            )

        video_id_to_source: Dict[str, int] = {}
        samples: List[Sample] = []

        for source_idx, (vdir, anno) in enumerate(zip(vdirs, annos)):
            ext = os.path.splitext(anno)[1].lower()
            if ext == ".csv":
                raw_pairs = _parse_csv(anno)
            else:
                raw_pairs = _detect_and_parse_json(anno, vdir, split_filter)

            per_video_count: Dict[str, int] = {}
            for vid, caption in raw_pairs:
                # Deduplication logic based on video_id across sources to avoid data leaks
                if vid in video_id_to_source and video_id_to_source[vid] != source_idx:
                    continue
                if vid not in video_id_to_source:
                    video_id_to_source[vid] = source_idx

                if captions_per_video is not None:
                    if per_video_count.get(vid, 0) >= captions_per_video:
                        continue
                    per_video_count[vid] = per_video_count.get(vid, 0) + 1

                path = _resolve_video_path(vdir, vid)
                if require_exists and not os.path.exists(path):
                    continue
                samples.append(Sample(video_path=path, caption=caption, video_id=vid))

        if limit is not None:
            samples = samples[: int(limit)]

        if not samples:
            raise RuntimeError(
                f"No samples found in {annotation_file!r} "
                f"(split_filter={split_filter!r}, video_dir={video_dir!r})."
            )
        self.samples: List[Sample] = samples

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def _decode_clip(self, path: str) -> Tensor:
        """Decode ``num_frames`` uniformly-sampled frames -> uint8 [T, C, H, W]."""
        from torchcodec.decoders import VideoDecoder  # lazy import

        decoder = VideoDecoder(path, device=self.decode_device)
        num_total = getattr(decoder.metadata, "num_frames", None)
        if not num_total:
            num_total = len(decoder)
        indices = _uniform_frame_indices(int(num_total), self.num_frames)

        # Decode only unique indices then gather, so repeated indices (short
        # clips) are cheap and version-independent.
        unique_sorted = sorted(set(indices))
        remap = {orig: i for i, orig in enumerate(unique_sorted)}
        batch = decoder.get_frames_at(indices=unique_sorted)
        decoded = batch.data  # uint8 [len(unique), C, H, W]
        gather_idx = torch.tensor([remap[i] for i in indices], dtype=torch.long)
        frames = decoded.index_select(0, gather_idx.to(decoded.device))
        return self._resize(frames)

    def _resize(self, frames_u8: Tensor) -> Tensor:
        """Resize uint8 [T, C, H, W] frames to resolution x resolution."""
        if frames_u8.shape[-2] == self.resolution and frames_u8.shape[-1] == self.resolution:
            return frames_u8.contiguous().to(torch.uint8)
        x = frames_u8.float()
        x = F.interpolate(
            x,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return x.round_().clamp_(0, 255).to(torch.uint8)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        sample = self.samples[index]
        try:
            frames = self._decode_clip(sample.video_path)
        except Exception as exc:  # noqa: BLE001 - resilient to a few corrupt clips
            # Fall back to a different random sample so a handful of corrupt /
            # missing videos do not abort an entire epoch.
            alt = self._rng.randrange(len(self.samples))
            if alt == index:
                raise RuntimeError(
                    f"Failed to decode {sample.video_path!r}: {exc}"
                ) from exc
            return self.__getitem__(alt)
        return frames, sample.caption

    # ------------------------------------------------------------------ #
    # Construction from config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls,
        cfg,
        split: str,
        *,
        limit: Optional[int] = None,
        decode_device: str = "cpu",
    ) -> "MSRVTTVideoTextDataset":
        """Build a dataset for ``split`` ("train" / "eval") from a config.

        Reads ``cfg.num_frames`` / ``cfg.resolution`` plus the ``cfg.data``
        section, tolerating a few common key spellings.
        """
        from utils import cfg_get  # local import to keep the Dataset standalone

        num_frames = int(cfg_get(cfg, "num_frames", "data.num_frames", default=64))
        resolution = int(cfg_get(cfg, "resolution", "data.resolution", default=256))
        video_dir = cfg_get(
            cfg, "data.video_dir", "data.videos_dir", "data.video_root",
            "video_dir", default=None,
        )

        if split == "train":
            annotation = cfg_get(
                cfg, "data.annotation_train", "data.train_annotation",
                "data.train_json", "data.annotations.train", default=None,
            )
            split_filter = cfg_get(cfg, "data.train_split_filter", default=None)
            captions_per_video = cfg_get(cfg, "data.train_captions_per_video", default=None)
        elif split in ("eval", "val", "test"):
            annotation = cfg_get(
                cfg, "data.annotation_eval", "data.eval_annotation",
                "data.test_annotation", "data.eval_json", "data.annotations.eval",
                default=None,
            )
            split_filter = cfg_get(cfg, "data.eval_split_filter", default=None)
            captions_per_video = cfg_get(cfg, "data.eval_captions_per_video", default=1)
        else:
            raise ValueError(f"Unknown split={split!r}.")

        if video_dir is None or annotation is None:
            raise KeyError(
                "Could not locate the video directory / annotation path in the "
                f"config for split={split!r}. Expected something like "
                "cfg.data.video_dir and cfg.data.annotation_train/eval."
            )

        return cls(
            video_dir=video_dir,
            annotation_file=annotation,
            num_frames=num_frames,
            resolution=resolution,
            split_filter=split_filter,
            limit=limit,
            captions_per_video=captions_per_video,
            decode_device=decode_device,
            seed=int(cfg_get(cfg, "seed", default=0)),
        )


def collate_fn(
    batch: Sequence[Tuple[Tensor, str]],
) -> Tuple[List[Tensor], List[str]]:
    """Collate into ``(list_of_frame_tensors, list_of_captions)``.

    Frames are intentionally **not** stacked: ``VisionEncoder.encode`` takes a
    list of per-clip tensors.
    """
    frames = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    return frames, captions


def build_dataset(
    cfg,
    split: str,
    *,
    limit: Optional[int] = None,
    decode_device: str = "cpu",
) -> MSRVTTVideoTextDataset:
    """Convenience wrapper around :meth:`MSRVTTVideoTextDataset.from_config`."""
    return MSRVTTVideoTextDataset.from_config(
        cfg, split, limit=limit, decode_device=decode_device
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSR-VTT video/text dataset smoke test.")
    parser.add_argument("--video-dir", required=True, help="Directory of video files.")
    parser.add_argument(
        "--annotation", required=True, help="videodatainfo JSON or CSV annotation file."
    )
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--split-filter", default=None)
    parser.add_argument(
        "--captions-per-video", type=int, default=None, help="Cap captions per video."
    )
    parser.add_argument(
        "--limit", type=int, default=8, help="Tiny debug subset size (default: 8)."
    )
    parser.add_argument("--decode-device", default="cpu")
    return parser


def _main() -> None:
    args = _build_arg_parser().parse_args()
    dataset = MSRVTTVideoTextDataset(
        video_dir=args.video_dir,
        annotation_file=args.annotation,
        num_frames=args.num_frames,
        resolution=args.resolution,
        split_filter=args.split_filter,
        limit=args.limit,
        captions_per_video=args.captions_per_video,
        decode_device=args.decode_device,
        require_exists=True,
    )
    print(f"[dataset] {len(dataset)} samples")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    frames, captions = next(iter(loader))
    print(f"[dataset] batch: {len(frames)} clips, {len(captions)} captions")
    print(f"[dataset] clip[0] shape={tuple(frames[0].shape)} dtype={frames[0].dtype}")
    print(f"[dataset] caption[0]={captions[0]!r}")


if __name__ == "__main__":
    _main()
