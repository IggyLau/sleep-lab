"""Parsing and request-shape tests against the documented Google Health API v4 schema.
No network: the HTTP session is faked."""

import json
import time
import urllib.parse

import pandas as pd
import pytest

from sleeplab import google_health as gh
from sleeplab.ingest import fetch_nights
from sleeplab.night import Night

SLEEP = {
    "name": "users/me/dataTypes/sleep/dataPoints/abc",
    "sleep": {
        "interval": {"startTime": "2026-10-25T06:10:00Z", "endTime": "2026-10-25T13:40:00Z"},
        "type": "STAGES",
        "stages": [
            {"startTime": "2026-10-25T06:10:00Z", "endTime": "2026-10-25T06:30:00Z", "type": "AWAKE"},
            {"startTime": "2026-10-25T06:30:00Z", "endTime": "2026-10-25T07:30:00Z", "type": "LIGHT"},
            {"startTime": "2026-10-25T07:30:00Z", "endTime": "2026-10-25T08:10:00Z", "type": "DEEP"},
            {"startTime": "2026-10-25T08:10:00Z", "endTime": "2026-10-25T13:40:00Z", "type": "REM"},
        ],
        "metadata": {"mainSleep": True, "nap": False},
        "summary": {"minutesAsleep": "430"},
    },
}
NAP = {"sleep": {"interval": {"startTime": "2026-10-25T21:00:00Z", "endTime": "2026-10-25T21:20:00Z"}, "metadata": {"mainSleep": False, "nap": True}}}


def spo2(t, v):
    return {"oxygenSaturation": {"sampleTime": {"physicalTime": t, "utcOffset": "-25200s"}, "percentage": v}}


def hr(t, v):
    return {"heartRate": {"sampleTime": {"physicalTime": t}, "beatsPerMinute": str(v)}}


RHR = {"dailyRestingHeartRate": {"date": {"year": 2026, "month": 10, "day": 25}, "beatsPerMinute": "54"}}


def test_parsers():
    s = gh.parse_sleep([NAP, SLEEP])
    assert [x.main for x in s] == [True, False]
    assert s[0].wake_date("America/Los_Angeles") == "2026-10-25"
    assert list(s[0].stages.stage) == ["AWAKE", "LIGHT", "DEEP", "REM"]
    df = gh.parse_spo2([spo2("2026-10-25T07:01:00Z", 95.5), {"oxygenSaturation": {}}])
    assert len(df) == 1 and df.spo2.iloc[0] == 95.5 and str(df.time.dt.tz) == "UTC"
    assert gh.parse_heart_rate([hr("2026-10-25T07:01:00Z", 61)]).bpm.iloc[0] == 61
    assert gh.parse_resting_hr([RHR]) == {"2026-10-25": 54}


def test_filters():
    a, b = pd.Timestamp("2026-10-25T06:00Z"), pd.Timestamp("2026-10-25T14:00Z")
    assert gh.sample_filter("oxygen-saturation", a, b) == (
        'oxygen_saturation.sample_time.physical_time >= "2026-10-25T06:00:00Z" AND '
        'oxygen_saturation.sample_time.physical_time < "2026-10-25T14:00:00Z"'
    )
    assert gh.daily_filter("daily-resting-heart-rate", "2026-10-24", "2026-10-26").startswith('daily_resting_heart_rate.date >= "2026-10-24"')


def test_auth_url_has_pkce_offline_and_scopes():
    cfg = gh.OAuthConfig("cid.apps.googleusercontent.com", "secret")
    verifier, challenge = gh.pkce_pair()
    q = urllib.parse.parse_qs(urllib.parse.urlparse(gh.build_auth_url(cfg, "st", challenge)).query)
    assert q["code_challenge_method"] == ["S256"] and q["access_type"] == ["offline"]
    assert "googlehealth.sleep.readonly" in q["scope"][0]
    assert 43 <= len(verifier) <= 128


def test_redirect_state_is_checked():
    assert gh.parse_redirect("http://localhost:8765/callback?code=abc&state=s1", "s1") == "abc"
    with pytest.raises(gh.AuthError):
        gh.parse_redirect("http://localhost:8765/callback?code=abc&state=evil", "s1")


def test_access_token_refreshes(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"access_token": "old", "refresh_token": "r", "expires_at": time.time() - 5}))

    class R:
        status_code = 200

        def json(self):
            return {"access_token": "new", "expires_in": 3600}

    monkeypatch.setattr(gh.requests, "post", lambda *a, **k: R())
    assert gh.access_token(gh.OAuthConfig("c", "s"), tf) == "new"
    assert json.loads(tf.read_text())["refresh_token"] == "r"


class FakeSession:
    """Answers like the API, paging oxygen-saturation to check pageToken handling."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        dtype = url.split("/dataTypes/")[1].split("/")[0]
        params = params or {}
        if dtype == "sleep":
            body = {"dataPoints": [SLEEP, NAP]}
        elif dtype == "oxygen-saturation":
            body = (
                {"dataPoints": [spo2("2026-10-25T07:00:00Z", 96)], "nextPageToken": "p2"}
                if "pageToken" not in params
                else {"dataPoints": [spo2("2026-10-25T07:01:00Z", 94)]}
            )
        elif dtype == "heart-rate":
            body = {"dataPoints": [hr("2026-10-25T07:00:00Z", 58)]}
        else:
            body = {"dataPoints": [RHR]}

        class R:
            status_code = 200

            def json(self_inner):
                return body

        return R()


def test_fetch_nights_end_to_end_without_network(tmp_path):
    tz = "America/Los_Angeles"
    # snores already detected for that night must survive a fetch
    t = pd.Timestamp("2026-10-25T07:00Z")
    Night(date="2026-10-25", snores=pd.DataFrame({"start": [t], "end": [t + pd.Timedelta(seconds=2)], "score": [0.8]})).save(tmp_path)

    sess = FakeSession()
    nights = fetch_nights(gh.HealthClient("tok", session=sess), tmp_path, "2026-10-25", "2026-10-25", tz)
    assert [n.date for n in nights] == ["2026-10-25"]
    n = Night.load(tmp_path, "2026-10-25")
    assert len(n.spo2) == 2 and n.meta["resting_hr"] == 54 and len(n.snores) == 1
    assert (tmp_path / "nights" / "2026-10-25" / "raw" / "oxygen-saturation.json").exists()
    assert sess.headers["Authorization"] == "Bearer tok"
    assert any("pageToken" in p for _, p in sess.calls)
