"""One night of data, and how it lives on disk.

Everything is stored as small CSVs under ``<data>/nights/<YYYY-MM-DD>/`` where the
date is the *wake-up* date in your local timezone (the same convention Fitbit uses).
All timestamps are UTC, tz-aware.

    stages.csv        start,end,stage           (AWAKE/LIGHT/DEEP/REM/ASLEEP/RESTLESS)
    spo2.csv          time,spo2                  (percent, ~1/min, gaps expected)
    heart_rate.csv    time,bpm
    snore_events.csv  start,end,score            (from the audio detector)
    meta.json         sleep window, resting HR, audio coverage, sources
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STAGES = ["AWAKE", "REM", "LIGHT", "DEEP", "ASLEEP", "RESTLESS"]

_COLUMNS = {
    "stages": ["start", "end", "stage"],
    "spo2": ["time", "spo2"],
    "heart_rate": ["time", "bpm"],
    "snores": ["start", "end", "score"],
}
_FILES = {
    "stages": "stages.csv",
    "spo2": "spo2.csv",
    "heart_rate": "heart_rate.csv",
    "snores": "snore_events.csv",
}
_TIME_COLS = {"start", "end", "time"}


def empty(kind: str) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in _COLUMNS[kind]})
    for c in _COLUMNS[kind]:
        if c in _TIME_COLS:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def _normalise(kind: str, df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty(kind)
    df = df[_COLUMNS[kind]].copy()
    for c in df.columns:
        if c in _TIME_COLS:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    sort_col = "time" if "time" in df.columns else "start"
    return df.sort_values(sort_col).reset_index(drop=True)


@dataclass
class Night:
    date: str
    stages: pd.DataFrame = field(default_factory=lambda: empty("stages"))
    spo2: pd.DataFrame = field(default_factory=lambda: empty("spo2"))
    heart_rate: pd.DataFrame = field(default_factory=lambda: empty("heart_rate"))
    snores: pd.DataFrame = field(default_factory=lambda: empty("snores"))
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind in _COLUMNS:
            setattr(self, kind, _normalise(kind, getattr(self, kind)))

    # ---- sleep window -------------------------------------------------
    @property
    def sleep_start(self) -> pd.Timestamp | None:
        if "sleep_start" in self.meta:
            return pd.Timestamp(self.meta["sleep_start"])
        if not self.stages.empty:
            return self.stages["start"].min()
        return None

    @property
    def sleep_end(self) -> pd.Timestamp | None:
        if "sleep_end" in self.meta:
            return pd.Timestamp(self.meta["sleep_end"])
        if not self.stages.empty:
            return self.stages["end"].max()
        return None

    @property
    def audio_window(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        a = self.meta.get("audio")
        if not a:
            return None
        return pd.Timestamp(a["start"]), pd.Timestamp(a["end"])

    # ---- persistence ----------------------------------------------------
    def save(self, root: Path) -> Path:
        d = Path(root) / "nights" / self.date
        d.mkdir(parents=True, exist_ok=True)
        for kind, fname in _FILES.items():
            df = getattr(self, kind)
            out = df.copy()
            for c in out.columns:
                if c in _TIME_COLS:
                    out[c] = out[c].map(lambda t: t.isoformat())
            out.to_csv(d / fname, index=False)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, root: Path, date: str) -> "Night":
        d = Path(root) / "nights" / date
        kwargs = {}
        for kind, fname in _FILES.items():
            p = d / fname
            kwargs[kind] = pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None
        meta_p = d / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        return cls(date=date, meta=meta, **kwargs)

    @classmethod
    def load_or_new(cls, root: Path, date: str) -> "Night":
        if (Path(root) / "nights" / date).exists():
            return cls.load(root, date)
        return cls(date=date)


def list_nights(root: Path) -> list[str]:
    d = Path(root) / "nights"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and len(p.name) == 10)
