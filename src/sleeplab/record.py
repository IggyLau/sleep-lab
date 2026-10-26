"""Record a night from the laptop mic, detecting snores as it goes.

Audio is captured in one-minute chunks. Each chunk is scored, its snore events
are appended to the night's CSV, and the chunk is thrown away. With
``max_clips`` > 0, a few short clips around the loudest snores are kept so you
can check (or later label) them. The full night is only written to disk if you
pass ``keep_audio`` explicitly.
"""

from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .audio import SR, Detector, events_from_scores, merge_events, to_absolute
from .night import Night


def record_night(
    data_root: Path,
    detector: Detector,
    tz: str,
    threshold: float = 0.3,
    chunk_s: float = 60.0,
    max_clips: int = 20,
    keep_audio: Path | None = None,
    max_hours: float = 11.0,
) -> Night:
    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()

    def _cb(indata, frames, t, status):  # runs on the audio thread: do nothing heavy
        q.put(indata[:, 0].copy())

    started = pd.Timestamp.now(tz="UTC")
    all_events: list[pd.DataFrame] = []
    clips_kept = 0
    clip_dir = None
    keeper = sf.SoundFile(str(keep_audio), "w", SR, 1) if keep_audio else None
    buf: list[np.ndarray] = []
    buffered = 0
    offset = 0.0
    print(f"Recording at {SR} Hz with the {detector.name} detector. Ctrl-C to stop (or it stops after {max_hours} h).")
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=_cb):
            while (pd.Timestamp.now(tz="UTC") - started).total_seconds() < max_hours * 3600:
                block = q.get()
                buf.append(block)
                buffered += len(block)
                if keeper:
                    keeper.write(block)
                if buffered < chunk_s * SR:
                    continue
                chunk = np.concatenate(buf)
                buf, buffered = [], 0
                ev = events_from_scores(detector.frame_scores(chunk), threshold, offset_s=offset)
                if not ev.empty:
                    all_events.append(ev)
                    if clips_kept < max_clips:
                        clip_dir = clip_dir or Path(data_root) / "clips" / started.strftime("%Y%m%dT%H%M%SZ")
                        clip_dir.mkdir(parents=True, exist_ok=True)
                        best = ev.nlargest(1, "score").iloc[0]
                        s = max(0, int((best.start_s - offset - 1.5) * SR))
                        e = int((best.end_s - offset + 1.5) * SR)
                        sf.write(str(clip_dir / f"snore_{best.start_s:08.1f}s_{best.score:.2f}.wav"), chunk[s:e], SR)
                        clips_kept += 1
                offset += len(chunk) / SR
                n = sum(len(x) for x in all_events)
                print(f"\r{offset / 60:6.0f} min recorded, {n} snore events so far", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if keeper:
            keeper.close()

    end = started + pd.Timedelta(seconds=offset)
    events = merge_events(pd.concat(all_events, ignore_index=True)) if all_events else events_from_scores(np.array([]))
    return attach_snores(data_root, events, started, end, detector.name, tz)


def night_date_for(end: pd.Timestamp, tz: str) -> str:
    """A night belongs to the local date you woke up on."""
    return pd.Timestamp(end).tz_convert(tz).strftime("%Y-%m-%d")


def attach_snores(
    data_root: Path, events_s: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, detector: str, tz: str
) -> Night:
    """Write detected events into the night they belong to, keeping any Fitbit data there."""
    night = Night.load_or_new(data_root, night_date_for(end, tz))
    night.snores = to_absolute(events_s, start)
    night.meta["audio"] = {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(), "detector": detector}
    night.meta.setdefault("timezone", tz)
    night.__post_init__()
    night.save(data_root)
    return night
