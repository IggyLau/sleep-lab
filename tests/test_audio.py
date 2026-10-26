import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sleeplab.audio import SR, EnergyDetector, detect_file, events_from_scores, save_clips
from sleeplab.record import attach_snores
from sleeplab.synthetic import snore_audio


def test_events_from_scores_merges_adjacent_frames():
    scores = np.array([0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.7, 0.0])
    ev = events_from_scores(scores, threshold=0.5)
    assert len(ev) == 2
    assert ev.start_s.iloc[0] == pytest.approx(0.48) and ev.end_s.iloc[0] == pytest.approx(0.96 + 0.96)
    assert ev.score.iloc[0] == pytest.approx(0.9)


def test_energy_detector_finds_bursts_in_file_with_chunking(tmp_path):
    at = [3.0, 7.5, 12.0, 21.0]
    p = tmp_path / "night.wav"
    sf.write(p, snore_audio(30, snore_at=at), SR)
    # small chunks so events must survive chunk boundaries
    ev, dur = detect_file(p, EnergyDetector(), threshold=0.5, chunk_s=10.0)
    assert dur == pytest.approx(30, abs=0.01)
    assert len(ev) == len(at)
    for (s, e), t in zip(ev[["start_s", "end_s"]].to_numpy(), at):
        assert s - 1.0 <= t <= e

    clips = save_clips(p, ev, tmp_path / "clips", top_n=2)
    assert len(clips) == 2 and all(c.stat().st_size > 0 for c in clips)


def test_attach_snores_assigns_wake_up_date_and_keeps_fitbit_data(tmp_path):
    tz = "America/Los_Angeles"
    start = pd.Timestamp("2026-10-24 23:30", tz=tz)
    ev = pd.DataFrame({"start_s": [10.0], "end_s": [11.0], "score": [0.9]})
    n = attach_snores(tmp_path, ev, start, start + pd.Timedelta(hours=7), "energy", tz)
    assert n.date == "2026-10-25"
    assert n.snores.start.iloc[0] == (start + pd.Timedelta(seconds=10)).tz_convert("UTC")
    assert n.meta["audio"]["detector"] == "energy"


@pytest.mark.skipif(not os.environ.get("SLEEPLAB_TEST_YAMNET"), reason="set SLEEPLAB_TEST_YAMNET=1 (downloads YAMNet)")
def test_yamnet_runs_and_is_quiet_on_silence():
    from sleeplab.audio import YamnetDetector

    det = YamnetDetector()
    scores = det.frame_scores(np.zeros(SR * 3, dtype=np.float32))
    assert scores.shape[0] >= 5 and scores.max() < 0.3
