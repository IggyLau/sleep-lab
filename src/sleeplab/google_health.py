"""Google Health API (the successor to the Fitbit Web API) for a Fitbit Air.

Auth is standard Google OAuth 2.0 (authorization code + PKCE) for *your own*
Google Cloud project in "Testing" mode with you as the only test user. Your
Google password is only ever typed into Google's own page in your browser; this
code only sees the authorization code and the tokens, which are stored in
``.tokens.json`` (gitignored, chmod 600).

API facts this module relies on (checked against developers.google.com/health, Oct 2026):
  base URL         https://health.googleapis.com/v4
  list points      GET /users/me/dataTypes/{data-type}/dataPoints?filter=...&pageToken=...
  data types       sleep (session), oxygen-saturation (1-min samples, sparse),
                   heart-rate (samples), daily-resting-heart-rate (daily)
  filters          {data_type}.sample_time.physical_time >= "RFC3339" AND ... < "..."
                   sleep.interval.end_time / sleep.interval.civil_end_time
                   {data_type}.date >= "YYYY-MM-DD"
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://health.googleapis.com/v4"

SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
    # SpO2, heart rate and resting heart rate are vitals. Google's scope page doesn't
    # spell out which scope covers them; these two are the candidates.
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
]

TOKEN_FILE = Path(".tokens.json")


class AuthError(RuntimeError):
    pass


# --------------------------------------------------------------------------- auth


@dataclass
class OAuthConfig:
    client_id: str
    client_secret: str
    redirect_uri: str = "http://localhost:8765/callback"

    @classmethod
    def from_env(cls) -> "OAuthConfig":
        cid = os.environ.get("GOOGLE_HEALTH_CLIENT_ID", "").strip()
        secret = os.environ.get("GOOGLE_HEALTH_CLIENT_SECRET", "").strip()
        if not cid or not secret:
            raise AuthError(
                "GOOGLE_HEALTH_CLIENT_ID / GOOGLE_HEALTH_CLIENT_SECRET are not set. "
                "Copy .env.example to .env and fill them in (see README: 'Google Health API setup')."
            )
        return cls(cid, secret, os.environ.get("GOOGLE_HEALTH_REDIRECT_URI", cls.redirect_uri).strip())


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(cfg: OAuthConfig, state: str, challenge: str, scopes: list[str] = SCOPES) -> str:
    q = {
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",  # we want a refresh token
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(q)}"


def _wait_for_redirect(redirect_uri: str, timeout_s: float = 300) -> str:
    """Serve one request on the loopback redirect URI and return the full URL it hit."""
    u = urllib.parse.urlparse(redirect_uri)
    got: dict[str, str] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            got["url"] = f"{redirect_uri.split('?')[0]}?{urllib.parse.urlparse(self.path).query}"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>Sleep Lab is authorised.</h2><p>You can close this tab and go back to the terminal.</p>")

        def log_message(self, *args):
            pass

    srv = http.server.HTTPServer((u.hostname or "localhost", u.port or 80), Handler)
    srv.timeout = 1
    deadline = time.time() + timeout_s
    while "url" not in got and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    if "url" not in got:
        raise AuthError("Timed out waiting for Google to redirect back.")
    return got["url"]


def parse_redirect(url: str, expected_state: str) -> str:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "error" in q:
        raise AuthError(f"Google returned an error: {q['error'][0]}")
    if q.get("state", [""])[0] != expected_state:
        raise AuthError("OAuth state mismatch; refusing the code (possible CSRF). Run `sleeplab auth` again.")
    if "code" not in q:
        raise AuthError("No ?code= in the redirect URL.")
    return q["code"][0]


def authorize(cfg: OAuthConfig, token_file: Path = TOKEN_FILE, open_browser: bool = True) -> dict:
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    url = build_auth_url(cfg, state, challenge)
    host = urllib.parse.urlparse(cfg.redirect_uri).hostname
    print("Opening Google's consent page. Sign in there (not here) and allow access.\n")
    print(url, "\n")
    if open_browser:
        webbrowser.open(url)
    if host in ("localhost", "127.0.0.1"):
        redirect = _wait_for_redirect(cfg.redirect_uri)
    else:
        redirect = input("After allowing, paste the full URL your browser ended up on:\n> ").strip()
    code = parse_redirect(redirect, state)
    r = requests.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
            "redirect_uri": cfg.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": verifier,
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise AuthError(f"Token exchange failed ({r.status_code}): {r.text}")
    tok = r.json()
    tok["expires_at"] = time.time() + tok.get("expires_in", 3600) - 60
    save_tokens(tok, token_file)
    return tok


def save_tokens(tok: dict, token_file: Path = TOKEN_FILE) -> None:
    token_file.write_text(json.dumps(tok, indent=2))
    token_file.chmod(0o600)


def access_token(cfg: OAuthConfig, token_file: Path = TOKEN_FILE) -> str:
    if not token_file.exists():
        raise AuthError("Not authorised yet. Run `sleeplab auth`.")
    tok = json.loads(token_file.read_text())
    if tok.get("expires_at", 0) > time.time():
        return tok["access_token"]
    if "refresh_token" not in tok:
        raise AuthError("Access token expired and there is no refresh token. Run `sleeplab auth`.")
    r = requests.post(
        TOKEN_URL,
        data={
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
            "refresh_token": tok["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=30,
    )
    if r.status_code != 200:
        # In Testing mode Google expires refresh tokens after 7 days.
        raise AuthError(f"Token refresh failed ({r.status_code}): {r.text}\nRun `sleeplab auth` again (Testing-mode tokens last 7 days).")
    new = r.json()
    tok.update(new)
    tok["expires_at"] = time.time() + new.get("expires_in", 3600) - 60
    save_tokens(tok, token_file)
    return tok["access_token"]


# --------------------------------------------------------------------------- client


class HealthClient:
    def __init__(self, token: str, base: str = API_BASE, session: requests.Session | None = None):
        self.base = base
        self.s = session or requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.s.get(f"{self.base}{path}", params=params, timeout=60)
        if r.status_code == 401:
            raise AuthError("401 from Google Health API. Run `sleeplab auth`.")
        if r.status_code >= 400:
            raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def identity(self) -> dict:
        return self._get("/users/me/identity")

    def paired_devices(self) -> dict:
        return self._get("/users/me/pairedDevices")

    def list_points(self, data_type: str, filter: str | None = None, page_size: int | None = None) -> Iterator[dict]:
        params: dict = {}
        if filter:
            params["filter"] = filter
        if page_size:
            params["pageSize"] = page_size
        while True:
            body = self._get(f"/users/me/dataTypes/{data_type}/dataPoints", params)
            yield from body.get("dataPoints", [])
            token = body.get("nextPageToken")
            if not token:
                return
            params["pageToken"] = token


def _iso(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_filter(data_type: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    f = data_type.replace("-", "_")
    return f'{f}.sample_time.physical_time >= "{_iso(start)}" AND {f}.sample_time.physical_time < "{_iso(end)}"'


def sleep_filter(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f'sleep.interval.end_time >= "{_iso(start)}" AND sleep.interval.end_time < "{_iso(end)}"'


def daily_filter(data_type: str, first: str, last_exclusive: str) -> str:
    f = data_type.replace("-", "_")
    return f'{f}.date >= "{first}" AND {f}.date < "{last_exclusive}"'


# --------------------------------------------------------------------------- parsing


def _camel(data_type: str) -> str:
    head, *rest = data_type.split("-")
    return head + "".join(w.title() for w in rest)


def parse_samples(points: list[dict], data_type: str, value_field: str, out_col: str) -> pd.DataFrame:
    key = _camel(data_type)
    rows = []
    for p in points:
        d = p.get(key) or {}
        t = (d.get("sampleTime") or {}).get("physicalTime")
        v = d.get(value_field)
        if t is None or v is None:
            continue
        rows.append((pd.Timestamp(t).tz_convert("UTC"), float(v)))
    return pd.DataFrame(rows, columns=["time", out_col]).sort_values("time").reset_index(drop=True)


def parse_spo2(points: list[dict]) -> pd.DataFrame:
    return parse_samples(points, "oxygen-saturation", "percentage", "spo2")


def parse_heart_rate(points: list[dict]) -> pd.DataFrame:
    df = parse_samples(points, "heart-rate", "beatsPerMinute", "bpm")
    df["bpm"] = df["bpm"].astype(int)
    return df


def parse_resting_hr(points: list[dict]) -> dict[str, int]:
    out = {}
    for p in points:
        d = p.get("dailyRestingHeartRate") or {}
        date = d.get("date") or {}
        if "beatsPerMinute" in d and date:
            out[f"{date['year']:04d}-{date['month']:02d}-{date['day']:02d}"] = int(d["beatsPerMinute"])
    return out


@dataclass
class SleepSession:
    start: pd.Timestamp
    end: pd.Timestamp
    main: bool
    nap: bool
    kind: str
    stages: pd.DataFrame
    summary: dict

    def wake_date(self, tz: str) -> str:
        return self.end.tz_convert(tz).strftime("%Y-%m-%d")


def parse_sleep(points: list[dict]) -> list[SleepSession]:
    sessions = []
    for p in points:
        s = p.get("sleep") or {}
        iv = s.get("interval") or {}
        if "startTime" not in iv or "endTime" not in iv:
            continue
        stages = pd.DataFrame(
            [
                (pd.Timestamp(st["startTime"]).tz_convert("UTC"), pd.Timestamp(st["endTime"]).tz_convert("UTC"), st.get("type", "SLEEP_STAGE_TYPE_UNSPECIFIED"))
                for st in s.get("stages", [])
                if "startTime" in st and "endTime" in st
            ],
            columns=["start", "end", "stage"],
        )
        md = s.get("metadata") or {}
        sessions.append(
            SleepSession(
                start=pd.Timestamp(iv["startTime"]).tz_convert("UTC"),
                end=pd.Timestamp(iv["endTime"]).tz_convert("UTC"),
                main=bool(md.get("mainSleep", True)),
                nap=bool(md.get("nap", False)),
                kind=s.get("type", "SLEEP_TYPE_UNSPECIFIED"),
                stages=stages.sort_values("start").reset_index(drop=True),
                summary=s.get("summary") or {},
            )
        )
    return sorted(sessions, key=lambda x: x.start)
