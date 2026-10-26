"""Pull nights from the Google Health API into ``data/nights/<date>/``.

For each main (non-nap) sleep session that ended in the date range, fetch SpO2
and heart rate covering that session (plus 30 minutes either side) and the
resting heart rate for the wake-up date. Raw API responses are saved next to the
parsed CSVs under ``raw/`` so you can see exactly what your Fitbit Air returns.
Snore events already in the night folder are left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import google_health as gh
from .night import Night

PAD = pd.Timedelta(minutes=30)


def fetch_nights(client: gh.HealthClient, data_root: Path, since: str, until: str, tz: str) -> list[Night]:
    """``since``/``until`` are wake-up dates (inclusive), local time."""
    lo = pd.Timestamp(since, tz=tz)
    hi = pd.Timestamp(until, tz=tz) + pd.Timedelta(days=1)

    sleep_raw = list(client.list_points("sleep", gh.sleep_filter(lo, hi)))
    sessions = [s for s in gh.parse_sleep(sleep_raw) if s.main and not s.nap]

    first = lo.strftime("%Y-%m-%d")
    last_excl = hi.strftime("%Y-%m-%d")
    try:
        rhr_raw = list(client.list_points("daily-resting-heart-rate", gh.daily_filter("daily-resting-heart-rate", first, last_excl)))
    except RuntimeError as e:  # don't lose the night over a missing daily metric
        print(f"  resting HR unavailable: {e}")
        rhr_raw = []
    resting = gh.parse_resting_hr(rhr_raw)

    nights = []
    for sess in sessions:
        date = sess.wake_date(tz)
        a, b = sess.start - PAD, sess.end + PAD
        spo2_raw = list(client.list_points("oxygen-saturation", gh.sample_filter("oxygen-saturation", a, b)))
        hr_raw = list(client.list_points("heart-rate", gh.sample_filter("heart-rate", a, b)))

        night = Night.load_or_new(data_root, date)
        night.stages = sess.stages
        night.spo2 = gh.parse_spo2(spo2_raw)
        night.heart_rate = gh.parse_heart_rate(hr_raw)
        night.meta.update(
            {
                "source": "google-health-api",
                "timezone": tz,
                "sleep_start": sess.start.isoformat(),
                "sleep_end": sess.end.isoformat(),
                "sleep_type": sess.kind,
                "sleep_summary": sess.summary,
                "resting_hr": resting.get(date),
            }
        )
        night.__post_init__()
        d = night.save(data_root)
        raw = d / "raw"
        raw.mkdir(exist_ok=True)
        session_raw = [p for p in sleep_raw if (p.get("sleep") or {}).get("interval", {}).get("startTime") and pd.Timestamp(p["sleep"]["interval"]["startTime"]) == sess.start]
        for name, payload in [("sleep", session_raw), ("oxygen-saturation", spo2_raw), ("heart-rate", hr_raw), ("daily-resting-heart-rate", rhr_raw)]:
            (raw / f"{name}.json").write_text(json.dumps(payload, indent=1))
        print(f"  {date}: {len(night.stages)} stage segments, {len(night.spo2)} SpO2 samples, {len(night.heart_rate)} HR samples, resting HR {night.meta['resting_hr']}")
        nights.append(night)
    if not sessions:
        print(f"  no main sleep sessions found between {since} and {until}")
    return nights
