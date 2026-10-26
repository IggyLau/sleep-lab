"""Snore detection on audio. Everything here runs locally; nothing is uploaded.

The detectors score short frames (0.96 s windows, 0.48 s hop, YAMNet's native
framing). ``events_from_scores`` turns frame scores into snore events. Audio is
processed in chunks, so a full night never has to sit in memory, and only event
timestamps (plus, if you ask, a few seconds-long clips) are written to disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

SR = 16000
HOP_S = 0.48
WIN_S = 0.96


class Detector(Protocol):
    name: str

    def frame_scores(self, wave: np.ndarray) -> np.ndarray:
        """Per-frame snore score in [0, 1] for 16 kHz mono float32 audio.
        Frame i covers [i*HOP_S, i*HOP_S + WIN_S)."""


class YamnetDetector:
    """Google's YAMNet (AudioSet, 521 classes), reading the "Snoring" class.

    Pretrained, no fine-tuning. Needs ``pip install -e '.[yamnet]'``; the model
    (~15 MB) downloads from TF Hub on first use and is cached by tensorflow_hub.
    """

    name = "yamnet"
    MODEL_URL = "https://tfhub.dev/google/yamnet/1"

    def __init__(self) -> None:
        if "SSL_CERT_FILE" not in os.environ:
            # python.org macOS builds ship without a CA bundle; borrow certifi's.
            try:
                import certifi

                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                pass
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import csv

        import tensorflow as tf
        import tensorflow_hub as hub

        self._tf = tf
        self.model = hub.load(self.MODEL_URL)
        with tf.io.gfile.GFile(self.model.class_map_path().numpy()) as f:
            names = [row["display_name"] for row in csv.DictReader(f)]
        self.class_index = names.index("Snoring")

    def frame_scores(self, wave: np.ndarray) -> np.ndarray:
        if len(wave) < int(WIN_S * SR):
            wave = np.pad(wave, (0, int(WIN_S * SR) - len(wave)))
        scores, _, _ = self.model(self._tf.constant(wave, dtype=self._tf.float32))
        return scores.numpy()[:, self.class_index]


class EnergyDetector:
    """Crude fallback: loud, low-frequency (60-800 Hz) sound relative to the room.

    It is *not* snore-specific (a passing truck counts). It exists so the pipeline
    runs without TensorFlow, and as a sanity baseline for YAMNet.
    """

    name = "energy"

    def __init__(self, db_above_floor: float = 12.0) -> None:
        self.db_above_floor = db_above_floor
        self._sos = signal.butter(4, [60, 800], btype="bandpass", fs=SR, output="sos")

    def frame_scores(self, wave: np.ndarray) -> np.ndarray:
        y = signal.sosfilt(self._sos, wave)
        hop, win = int(HOP_S * SR), int(WIN_S * SR)
        n = max(1, 1 + (len(y) - win) // hop)
        rms = np.array([np.sqrt(np.mean(y[i * hop : i * hop + win] ** 2) + 1e-12) for i in range(n)])
        db = 20 * np.log10(rms)
        floor = np.percentile(db, 20)
        return 1 / (1 + np.exp(-(db - floor - self.db_above_floor) / 2.0))


def get_detector(name: str) -> Detector:
    if name == "yamnet":
        return YamnetDetector()
    if name == "energy":
        return EnergyDetector()
    raise ValueError(f"unknown detector {name!r} (choose yamnet or energy)")


def events_from_scores(
    scores: np.ndarray, threshold: float = 0.3, merge_gap_s: float = 0.5, offset_s: float = 0.0
) -> pd.DataFrame:
    """Frames above ``threshold`` become events; events closer than ``merge_gap_s`` merge.
    Returns start_s, end_s (seconds, relative to the audio start + offset) and peak score."""
    hot = scores >= threshold
    rows: list[list[float]] = []
    for i in np.flatnonzero(hot):
        s, e = offset_s + i * HOP_S, offset_s + i * HOP_S + WIN_S
        if rows and s - rows[-1][1] <= merge_gap_s:
            rows[-1][1] = e
            rows[-1][2] = max(rows[-1][2], float(scores[i]))
        else:
            rows.append([s, e, float(scores[i])])
    return pd.DataFrame(rows, columns=["start_s", "end_s", "score"])


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read any libsndfile format (wav, flac, ogg, mp3). For m4a: convert with ffmpeg first."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return y.astype(np.float32)
    g = np.gcd(sr, SR)
    return signal.resample_poly(y, SR // g, sr // g).astype(np.float32)


def iter_chunks(path: Path, chunk_s: float = 600.0) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (offset_seconds, 16 kHz mono chunk) without loading the whole file."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        block = int(chunk_s * sr)
        offset = 0.0
        while True:
            y = f.read(block, dtype="float32", always_2d=True)
            if len(y) == 0:
                break
            yield offset, to_16k(y.mean(axis=1), sr)
            offset += len(y) / sr


def merge_events(df: pd.DataFrame, merge_gap_s: float = 0.5) -> pd.DataFrame:
    """Merge events across chunk boundaries."""
    if df.empty:
        return df
    df = df.sort_values("start_s").reset_index(drop=True)
    rows = [df.iloc[0].tolist()]
    for s, e, sc in df.itertuples(index=False):
        if s - rows[-1][1] <= merge_gap_s:
            rows[-1][1] = max(rows[-1][1], e)
            rows[-1][2] = max(rows[-1][2], sc)
        else:
            rows.append([s, e, sc])
    return pd.DataFrame(rows, columns=["start_s", "end_s", "score"])


def detect_file(
    path: Path, detector: Detector, threshold: float = 0.3, chunk_s: float = 600.0
) -> tuple[pd.DataFrame, float]:
    """Run ``detector`` over a file in chunks. Returns (events in seconds, duration_s)."""
    parts, duration = [], 0.0
    for offset, chunk in iter_chunks(path, chunk_s):
        parts.append(events_from_scores(detector.frame_scores(chunk), threshold, offset_s=offset))
        duration = offset + len(chunk) / SR
    events = pd.concat(parts, ignore_index=True) if parts else events_from_scores(np.array([]))
    return merge_events(events), duration


def to_absolute(events: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(start).tz_convert("UTC")
    return pd.DataFrame(
        {
            "start": start + pd.to_timedelta(events["start_s"], unit="s"),
            "end": start + pd.to_timedelta(events["end_s"], unit="s"),
            "score": events["score"].round(3),
        }
    )


def save_clips(
    path: Path, events: pd.DataFrame, out_dir: Path, top_n: int = 10, pad_s: float = 1.5
) -> list[Path]:
    """Keep the ``top_n`` highest-scoring snores as short wav clips (for listening /
    labelling later). This is the only audio that is kept unless you choose otherwise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        for _, ev in events.nlargest(top_n, "score").iterrows():
            s = max(0, int((ev.start_s - pad_s) * sr))
            e = int((ev.end_s + pad_s) * sr)
            f.seek(s)
            y = f.read(e - s, dtype="float32", always_2d=True).mean(axis=1)
            p = out_dir / f"snore_{ev.start_s:08.1f}s_{ev.score:.2f}.wav"
            sf.write(str(p), y, sr)
            written.append(p)
    return written
