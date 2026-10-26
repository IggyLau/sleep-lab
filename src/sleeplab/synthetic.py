"""Fake nights, so the whole pipeline runs (and is testable) without a wrist or a mic.

The generator is deliberately physiological-ish rather than random noise:

* a hypnogram of ~90-minute cycles (deep sleep front-loaded, REM back-loaded),
* snoring arrives in *bouts* of breath-paced snores, mostly in LIGHT/DEEP sleep,
* every bout drags SpO2 down a little, starting ~40 s after the bout begins and
  bottoming out ~1-2 min in (circulation delay + Fitbit's 1-minute averaging),
* SpO2 has random gaps, because the real Fitbit stream does too.

So the analysis has a real, known effect to find: snore bouts lead SpO2 dips by
roughly one minute. If the report can't recover that, the report is broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .night import Night


@dataclass(frozen=True)
class Profile:
    name: str
    bouts_per_hour: float  # snore bouts per hour of LIGHT/DEEP sleep
    dip_per_minute: float  # SpO2 drop (percentage points) per minute of snoring
    baseline: float = 95.8


PROFILES = [
    Profile("after pizza and a late beer", bouts_per_hour=4.5, dip_per_minute=0.9),
    Profile("a perfectly normal Tuesday", bouts_per_hour=1.6, dip_per_minute=0.6),
    Profile("slept on my side out of spite", bouts_per_hour=0.5, dip_per_minute=0.4),
]

_STAGE_SNORE_WEIGHT = {"LIGHT": 1.0, "DEEP": 1.3, "REM": 0.35, "AWAKE": 0.0}
_STAGE_HR_OFFSET = {"LIGHT": 0.0, "DEEP": -4.0, "REM": 3.0, "AWAKE": 8.0}


def _hypnogram(rng: np.random.Generator, start: pd.Timestamp, hours: float) -> pd.DataFrame:
    rows = []
    t = start
    end = start + pd.Timedelta(hours=hours)
    rows.append((t, t + pd.Timedelta(minutes=rng.uniform(8, 20)), "AWAKE"))  # falling asleep
    t = rows[-1][1]
    cycle = 0
    while t < end:
        deep = max(2.0, rng.normal(32 - 9 * cycle, 5))
        rem = max(4.0, rng.normal(8 + 7 * cycle, 3))
        plan = [("LIGHT", rng.normal(30, 6)), ("DEEP", deep), ("LIGHT", rng.normal(14, 4)), ("REM", rem)]
        if rng.random() < 0.45:
            plan.append(("AWAKE", rng.uniform(1, 5)))
        for stage, minutes in plan:
            seg_end = min(end, t + pd.Timedelta(minutes=max(1.0, minutes)))
            rows.append((t, seg_end, stage))
            t = seg_end
            if t >= end:
                break
        cycle += 1
    df = pd.DataFrame(rows, columns=["start", "end", "stage"])
    df["start"] = df["start"].dt.round("s")
    df["end"] = df["end"].dt.round("s")
    return df


def _stage_at(stages: pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    idx = np.searchsorted(stages["start"].values, times.values, side="right") - 1
    idx = np.clip(idx, 0, len(stages) - 1)
    return stages["stage"].values[idx]


def generate_night(date: str, profile: Profile | None = None, seed: int = 0, tz: str = "America/Los_Angeles") -> Night:
    """Generate one night that *ends* on the morning of ``date`` (local time)."""
    rng = np.random.default_rng(seed)
    profile = profile or PROFILES[seed % len(PROFILES)]
    wake_day = pd.Timestamp(date, tz=tz)
    bedtime = (wake_day - pd.Timedelta(days=1)).replace(hour=23) + pd.Timedelta(minutes=float(rng.uniform(-40, 40)))
    hours = float(rng.uniform(6.8, 8.0))
    stages = _hypnogram(rng, bedtime.tz_convert("UTC"), hours)
    start, end = stages["start"].iloc[0], stages["end"].iloc[-1]

    # ---- snore bouts -> individual snore events ------------------------------
    minutes = pd.date_range(start, end, freq="1min", inclusive="left")
    stage_per_min = _stage_at(stages, minutes)
    weights = np.array([_STAGE_SNORE_WEIGHT.get(s, 0.0) for s in stage_per_min])
    p_bout = profile.bouts_per_hour / 60.0 * weights
    bout_starts = minutes[rng.random(len(minutes)) < p_bout]

    events, bouts = [], []
    busy_until = start
    for b in bout_starts:
        if b < busy_until:
            continue
        b = b + pd.Timedelta(seconds=float(rng.uniform(0, 59)))
        dur = float(rng.uniform(1.0, 7.0))  # minutes
        b_end = min(end, b + pd.Timedelta(minutes=dur))
        bouts.append((b, b_end))
        t = b
        while t < b_end:
            snore_len = rng.uniform(0.9, 2.2)
            events.append((t, t + pd.Timedelta(seconds=snore_len), float(np.clip(rng.normal(0.7, 0.12), 0.31, 0.99))))
            t += pd.Timedelta(seconds=float(rng.normal(4.2, 0.6)))  # one snore per breath
        busy_until = b_end + pd.Timedelta(minutes=2)
    # a few false positives (traffic, the cat, the duvet)
    for _ in range(rng.poisson(6)):
        t = start + (end - start) * float(rng.random())
        events.append((t, t + pd.Timedelta(seconds=float(rng.uniform(0.5, 1.5))), float(rng.uniform(0.3, 0.5))))
    snores = pd.DataFrame(events, columns=["start", "end", "score"]).sort_values("start")

    # ---- SpO2: baseline + bout-locked dips + noise + gaps ----------------------
    t_min = (minutes - start).total_seconds().values / 60.0
    spo2 = profile.baseline + 0.25 * np.sin(t_min / 90 * 2 * np.pi) + rng.normal(0, 0.45, len(minutes))
    for b, b_end in bouts:
        dur_min = (b_end - b).total_seconds() / 60
        depth = profile.dip_per_minute * dur_min * rng.uniform(0.7, 1.3)
        onset = (b - start).total_seconds() / 60 + rng.uniform(0.5, 0.9)  # circulation delay
        x = t_min - onset
        shape = np.where(x < 0, 0.0, np.where(x < dur_min, x / dur_min, np.exp(-(x - dur_min) / 1.5)))
        spo2 -= depth * shape
    spo2 = np.clip(np.round(spo2, 1), 82, 100)
    keep = rng.random(len(minutes)) > 0.12
    spo2_df = pd.DataFrame({"time": minutes[keep], "spo2": spo2[keep]})

    # ---- heart rate -------------------------------------------------------------
    resting = float(rng.normal(56, 2))
    hr = resting + np.array([_STAGE_HR_OFFSET.get(s, 0) for s in stage_per_min]) + rng.normal(0, 1.8, len(minutes))
    hr_df = pd.DataFrame({"time": minutes, "bpm": np.round(hr).astype(int)})

    meta = {
        "source": "synthetic",
        "profile": profile.name,
        "seed": seed,
        "timezone": tz,
        "sleep_start": start.isoformat(),
        "sleep_end": end.isoformat(),
        "resting_hr": round(resting),
        "audio": {"start": start.isoformat(), "end": end.isoformat(), "detector": "synthetic"},
    }
    return Night(date=date, stages=stages, spo2=spo2_df, heart_rate=hr_df, snores=snores, meta=meta)


def generate_nights(n: int = 3, last_date: str = "2026-10-25", seed: int = 7, tz: str = "America/Los_Angeles") -> list[Night]:
    last = pd.Timestamp(last_date)
    nights = []
    for i in range(n):
        date = (last - pd.Timedelta(days=n - 1 - i)).strftime("%Y-%m-%d")
        nights.append(generate_night(date, PROFILES[i % len(PROFILES)], seed=seed + i, tz=tz))
    return nights


def snore_audio(seconds: float = 30.0, sr: int = 16000, snore_at: list[float] | None = None, seed: int = 0) -> np.ndarray:
    """A crude synthetic snore track: quiet room noise plus low, buzzy, breath-length bursts.

    Good enough to exercise the windowing / event code. YAMNet is not fooled by it
    and that's fine: the YAMNet path is checked against real audio, not this.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = rng.normal(0, 0.003, n).astype(np.float32)
    t = np.arange(int(1.4 * sr)) / sr
    env = np.sin(np.pi * t / t[-1]) ** 2
    f0 = 70.0
    buzz = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    burst = (0.25 * env * buzz * (1 + 0.3 * np.sin(2 * np.pi * 30 * t))).astype(np.float32)
    for s in snore_at or []:
        i = int(s * sr)
        j = min(n, i + len(burst))
        y[i:j] += burst[: j - i]
    return y
