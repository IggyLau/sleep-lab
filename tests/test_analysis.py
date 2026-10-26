import pandas as pd

from sleeplab.analysis import desaturations, event_triggered_average, lagged_xcorr, minute_grid, snore_bouts, summarise
from sleeplab.night import Night, list_nights
from sleeplab.synthetic import PROFILES, generate_night


def heavy_night():
    return generate_night("2026-10-23", PROFILES[0], seed=7)


def test_synthetic_effect_is_recovered():
    """The generator plants 'SpO2 dips ~1 min after snoring'; the analysis must find it."""
    n = heavy_night()
    g = minute_grid(n)
    xc = lagged_xcorr(g)
    pos = xc[xc.lag_min.between(0, 6)]
    neg = xc[xc.lag_min.between(-10, -6)]
    assert pos.r.min() < -0.4, "snoring should predict lower SpO2 shortly after"
    assert pos.r.min() < neg.r.min(), "the effect should follow snoring, not precede it"

    eta = event_triggered_average(g, snore_bouts(n.snores))
    lag, depth = eta.nadir
    assert eta.n_bouts >= 5
    assert 0 < lag <= 8 and depth < -0.8
    assert eta.beats_null


def test_quiet_night_has_no_effect():
    n = heavy_night()
    n.snores = n.snores.iloc[0:0]
    s = summarise(Night(date=n.date, stages=n.stages, spo2=n.spo2, heart_rate=n.heart_rate, snores=None, meta=n.meta))
    assert s["snore_bouts"] == 0 and s["snore_minutes"] == 0
    assert s["verdict"].startswith("Not that the microphone noticed")


def test_snore_bouts_group_breaths():
    t = pd.Timestamp("2026-10-23T07:00:00Z")
    rows = [(t + pd.Timedelta(seconds=4 * i), t + pd.Timedelta(seconds=4 * i + 1.5), 0.8) for i in range(10)]  # one bout
    rows += [(t + pd.Timedelta(minutes=10), t + pd.Timedelta(minutes=10, seconds=1), 0.5)]  # lone false positive
    b = snore_bouts(pd.DataFrame(rows, columns=["start", "end", "score"]))
    assert len(b) == 1 and b.n.iloc[0] == 10


def test_minute_grid_counts_snore_seconds_across_minute_boundary():
    t = pd.Timestamp("2026-10-23T07:00:00Z")
    n = Night(
        date="2026-10-23",
        stages=pd.DataFrame({"start": [t], "end": [t + pd.Timedelta(minutes=5)], "stage": ["LIGHT"]}),
        snores=pd.DataFrame({"start": [t + pd.Timedelta(seconds=50)], "end": [t + pd.Timedelta(seconds=70)], "score": [0.9]}),
    )
    g = minute_grid(n)
    assert list(g.snore_s.iloc[:3]) == [10.0, 10.0, 0.0]


def test_desaturations_merge_consecutive_minutes():
    idx = pd.date_range("2026-10-23T07:00Z", periods=12, freq="1min")
    g = pd.DataFrame({"spo2": [96, 96, 96, 96, 92, 91.5, 92, 96, 96, 96, 96, 96]}, index=idx)
    assert len(desaturations(g)) == 1


def test_night_roundtrip(tmp_path):
    n = heavy_night()
    n.save(tmp_path)
    assert list_nights(tmp_path) == ["2026-10-23"]
    m = Night.load(tmp_path, "2026-10-23")
    assert len(m.snores) == len(n.snores) and len(m.spo2) == len(n.spo2)
    assert m.spo2.time.dt.tz is not None
    assert summarise(m)["snore_bouts"] == summarise(n)["snore_bouts"]
