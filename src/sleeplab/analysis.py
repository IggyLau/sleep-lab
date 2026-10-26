"""Line the two signals up and ask whether snoring and SpO2 dips go together.

Everything is put on a 1-minute grid, because that is what Fitbit SpO2 gives us.
Two complementary questions:

1. **Lagged cross-correlation**: does "seconds of snoring in minute t" correlate
   with SpO2 at minute t + lag? A dip that *follows* snoring shows up as a
   negative r at a small positive lag.
2. **Event-triggered average**: line up every snore *bout* onset and average the
   SpO2 change around it. Compared against the same thing at random times
   (shuffled onsets), so "it dips" has to beat "SpO2 just wobbles".

Neither is a diagnosis. See the README's "not a medical device" note.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .night import Night


def minute_grid(night: Night) -> pd.DataFrame:
    start, end = night.sleep_start, night.sleep_end
    if start is None or end is None:
        raise ValueError(f"{night.date}: no sleep window (no stages and no sleep_start/sleep_end)")
    idx = pd.date_range(start.floor("min"), end.ceil("min"), freq="1min", inclusive="left")
    g = pd.DataFrame(index=idx)

    sp = night.spo2.set_index("time")["spo2"]
    g["spo2"] = sp.groupby(sp.index.floor("min")).mean().reindex(idx) if len(sp) else np.nan
    hr = night.heart_rate.set_index("time")["bpm"]
    g["bpm"] = hr.groupby(hr.index.floor("min")).mean().reindex(idx) if len(hr) else np.nan

    # seconds of snoring that fall inside each minute
    snore_s = np.zeros(len(idx))
    t0 = idx[0]
    for s, e in zip(night.snores["start"], night.snores["end"]):
        a = (s - t0).total_seconds()
        b = (e - t0).total_seconds()
        for m in range(max(0, int(a // 60)), min(len(idx), int(b // 60) + 1)):
            snore_s[m] += max(0.0, min(b, (m + 1) * 60) - max(a, m * 60))
    g["snore_s"] = snore_s

    g["stage"] = "UNKNOWN"
    for s, e, st in night.stages.itertuples(index=False):
        g.loc[(g.index >= s.floor("min")) & (g.index < e), "stage"] = st

    aw = night.audio_window
    g["audio"] = True if aw is None else (g.index >= aw[0].floor("min")) & (g.index < aw[1])
    return g


def snore_bouts(snores: pd.DataFrame, max_gap_s: float = 30.0, min_events: int = 3) -> pd.DataFrame:
    """Group individual snores (one per breath) into bouts."""
    if snores.empty:
        return pd.DataFrame(columns=["start", "end", "n"])
    rows = []
    cur = None
    for s, e in zip(snores["start"], snores["end"]):
        if cur and (s - cur[1]).total_seconds() <= max_gap_s:
            cur[1] = max(cur[1], e)
            cur[2] += 1
        else:
            if cur:
                rows.append(cur)
            cur = [s, e, 1]
    rows.append(cur)
    df = pd.DataFrame(rows, columns=["start", "end", "n"])
    return df[df["n"] >= min_events].reset_index(drop=True)


def lagged_xcorr(grid: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """Pearson r between snore_s(t) and spo2(t + lag), lag in minutes."""
    x = grid["snore_s"].where(grid["audio"])
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        y = grid["spo2"].shift(-lag)
        ok = x.notna() & y.notna()
        n = int(ok.sum())
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if n > 10 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan
        rows.append((lag, r, n))
    return pd.DataFrame(rows, columns=["lag_min", "r", "n"])


def _eta(spo2: np.ndarray, onsets: np.ndarray, pre: int, post: int) -> tuple[np.ndarray, int]:
    """Mean SpO2 change vs. each onset's own pre-onset baseline."""
    segs = []
    for o in onsets:
        if o - pre < 0 or o + post >= len(spo2):
            continue
        w = spo2[o - pre : o + post + 1]
        base = np.nanmean(w[:pre])
        if np.isnan(base):
            continue
        segs.append(w - base)
    if not segs:
        return np.full(pre + post + 1, np.nan), 0
    with warnings.catch_warnings():  # a lag where every segment has a SpO2 gap is fine: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(segs), axis=0), len(segs)


@dataclass
class TriggeredAverage:
    lags: np.ndarray  # minutes relative to bout onset
    mean: np.ndarray  # SpO2 change, percentage points
    null_lo: np.ndarray  # 2.5th percentile of shuffled onsets
    null_hi: np.ndarray  # 97.5th
    n_bouts: int

    @property
    def nadir(self) -> tuple[float, float]:
        post = self.lags >= 0
        if self.n_bouts == 0 or np.all(np.isnan(self.mean[post])):
            return np.nan, np.nan
        i = int(np.nanargmin(self.mean[post]))
        return float(self.lags[post][i]), float(self.mean[post][i])

    @property
    def beats_null(self) -> bool:
        lag, depth = self.nadir
        if np.isnan(depth):
            return False
        return bool(depth < self.null_lo[list(self.lags).index(lag)])


def event_triggered_average(
    grid: pd.DataFrame, bouts: pd.DataFrame, pre: int = 5, post: int = 10, n_shuffle: int = 300, seed: int = 0
) -> TriggeredAverage:
    spo2 = grid["spo2"].to_numpy(dtype=float)
    t0 = grid.index[0]
    onsets = np.array([int((b - t0).total_seconds() // 60) for b in bouts["start"]], dtype=int)
    mean, n = _eta(spo2, onsets, pre, post)
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(grid["audio"].to_numpy() & (grid["stage"] != "AWAKE").to_numpy())
    null = []
    if n and len(valid) > pre + post:
        for _ in range(n_shuffle):
            m, _k = _eta(spo2, rng.choice(valid, size=max(n, 1)), pre, post)
            null.append(m)
    null = np.vstack(null) if null else np.full((1, pre + post + 1), np.nan)
    return TriggeredAverage(
        lags=np.arange(-pre, post + 1),
        mean=mean,
        null_lo=np.nanpercentile(null, 2.5, axis=0),
        null_hi=np.nanpercentile(null, 97.5, axis=0),
        n_bouts=n,
    )


def desaturations(grid: pd.DataFrame, drop: float = 3.0, window: int = 5) -> pd.DatetimeIndex:
    """Minutes where SpO2 is >= ``drop`` points below the max of the preceding ``window`` minutes.
    Consecutive minutes count as one event. A rough, Fitbit-resolution cousin of ODI-3."""
    sp = grid["spo2"]
    ref = sp.rolling(window, min_periods=1).max().shift(1)
    hit = (ref - sp) >= drop
    starts = hit & ~hit.shift(1, fill_value=False)
    return grid.index[starts.to_numpy()]


def summarise(night: Night) -> dict:
    g = minute_grid(night)
    asleep = g["stage"].isin(["LIGHT", "DEEP", "REM", "ASLEEP", "RESTLESS"])
    sleep_h = asleep.sum() / 60
    bouts = snore_bouts(night.snores)
    xc = lagged_xcorr(g)
    pos = xc[xc["lag_min"] >= 0].dropna()
    best = pos.loc[pos["r"].idxmin()] if not pos.empty else None
    eta = event_triggered_average(g, bouts)
    lag, depth = eta.nadir
    desat = desaturations(g)
    snore_min = float((g["snore_s"] > 0).sum())
    by_stage = (
        g[g["snore_s"] > 0].groupby("stage").size().reindex(["LIGHT", "DEEP", "REM", "AWAKE"]).fillna(0)
        / g.groupby("stage").size().reindex(["LIGHT", "DEEP", "REM", "AWAKE"]).replace(0, np.nan)
        * 100
    )
    s = {
        "date": night.date,
        "source": night.meta.get("source", "unknown"),
        "profile": night.meta.get("profile"),
        "sleep_hours": round(float(sleep_h), 2),
        "snore_events": int(len(night.snores)),
        "snore_bouts": int(len(bouts)),
        "snore_minutes": snore_min,
        "snore_pct_of_sleep": round(100 * snore_min / max(1.0, asleep.sum()), 1),
        "bouts_per_hour": round(len(bouts) / max(sleep_h, 1e-9), 2) if sleep_h else 0.0,
        "spo2_mean": round(float(g["spo2"].mean()), 1) if g["spo2"].notna().any() else None,
        "spo2_min": round(float(g["spo2"].min()), 1) if g["spo2"].notna().any() else None,
        "spo2_coverage_pct": round(100 * g["spo2"].notna().mean(), 0),
        "desats_3pct": int(len(desat)),
        "desats_per_hour": round(len(desat) / max(sleep_h, 1e-9), 1) if sleep_h else 0.0,
        "resting_hr": night.meta.get("resting_hr"),
        "xcorr_best_lag_min": int(best["lag_min"]) if best is not None else None,
        "xcorr_best_r": round(float(best["r"]), 2) if best is not None else None,
        "eta_nadir_lag_min": None if np.isnan(lag) else int(lag),
        "eta_nadir_drop": None if np.isnan(depth) else round(depth, 2),
        "eta_beats_shuffle": eta.beats_null,
        "snore_pct_by_stage": {k: (None if np.isnan(v) else round(float(v), 1)) for k, v in by_stage.items()},
    }
    s["verdict"] = verdict(s)
    return s


def verdict(s: dict) -> str:
    bph = s["bouts_per_hour"]
    if bph >= 3:
        head = "Yes. Emphatically."
    elif bph >= 1:
        head = "Yes, a bit."
    elif bph > 0.2:
        head = "Barely. Occasional snore cameos."
    else:
        head = "Not that the microphone noticed."
    if s["eta_beats_shuffle"] and s["eta_nadir_drop"] is not None:
        tail = f" SpO2 dips ~{abs(s['eta_nadir_drop']):.1f} points about {s['eta_nadir_lag_min']} min after a bout starts, more than chance."
    elif s["snore_bouts"]:
        tail = " SpO2 doesn't move more after snoring than at random times."
    else:
        tail = ""
    return head + tail
