"""sleeplab: the command line.

    sleeplab demo                  synthetic nights -> reports/demo/index.html (no accounts, no mic)
    sleeplab auth                  one-time Google sign-in (opens your browser)
    sleeplab whoami                check the token works and list paired devices
    sleeplab fetch --since DATE    pull sleep, SpO2, heart rate, resting HR into data/
    sleeplab detect FILE --start   find snores in an audio file you recorded
    sleeplab record                record tonight from the laptop mic (snores only are kept)
    sleeplab report                build reports/index.html from everything in data/
    sleeplab cover                 render the showcase cover chart from a synthetic night
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .night import Night, list_nights


def _tz() -> str:
    return os.environ.get("SLEEPLAB_TZ", "America/Los_Angeles")


def cmd_demo(a) -> None:
    from .report import build_report
    from .synthetic import generate_nights

    nights = generate_nights(a.nights, last_date=a.last_date, seed=a.seed, tz=_tz())
    for n in nights:
        n.save(a.data)
    idx = build_report(nights, a.out)
    print(f"{len(nights)} synthetic nights -> {a.data}/nights/\nReport: {idx.resolve()}")


def cmd_auth(a) -> None:
    from . import google_health as gh

    cfg = gh.OAuthConfig.from_env()
    gh.authorize(cfg, open_browser=not a.no_browser)
    print(f"Saved tokens to {gh.TOKEN_FILE.resolve()} (gitignored). Next: sleeplab whoami")


def _client():
    from . import google_health as gh

    cfg = gh.OAuthConfig.from_env()
    return gh.HealthClient(gh.access_token(cfg))


def cmd_whoami(a) -> None:
    c = _client()
    print(json.dumps(c.identity(), indent=2))
    try:
        print(json.dumps(c.paired_devices(), indent=2))
    except RuntimeError as e:
        print(f"pairedDevices: {e}")


def cmd_fetch(a) -> None:
    from .ingest import fetch_nights

    until = a.until or pd.Timestamp.now(tz=_tz()).strftime("%Y-%m-%d")
    print(f"Fetching nights {a.since} .. {until} ({_tz()})")
    fetch_nights(_client(), a.data, a.since, until, _tz())


def cmd_detect(a) -> None:
    from .audio import detect_file, get_detector, save_clips
    from .record import attach_snores

    start = pd.Timestamp(a.start)
    if start.tzinfo is None:
        start = start.tz_localize(_tz())
    det = get_detector(a.detector)
    print(f"Scoring {a.file} with {det.name} (threshold {a.threshold}) ...")
    events, dur = detect_file(a.file, det, threshold=a.threshold)
    end = start + pd.Timedelta(seconds=dur)
    night = attach_snores(a.data, events, start, end, det.name, _tz())
    print(f"{len(events)} snore events in {dur / 3600:.1f} h of audio -> night {night.date}")
    if a.clips:
        paths = save_clips(a.file, events, Path(a.data) / "nights" / night.date / "clips", top_n=a.clips)
        print(f"Kept {len(paths)} short clips for listening/labelling.")
    print("The source file was not modified or copied. Delete it when you're done, if you like.")


def cmd_record(a) -> None:
    from .audio import get_detector
    from .record import record_night

    night = record_night(a.data, get_detector(a.detector), _tz(), threshold=a.threshold,
                         max_clips=a.clips, keep_audio=a.keep_audio, max_hours=a.max_hours)
    print(f"{len(night.snores)} snore events saved to night {night.date}. Next: sleeplab fetch --since {night.date}")


def cmd_report(a) -> None:
    from .report import build_report

    dates = list_nights(a.data)
    if not dates:
        sys.exit(f"No nights in {a.data}/nights. Try `sleeplab demo`, or fetch + detect first.")
    nights = []
    for d in dates:
        n = Night.load(a.data, d)
        if n.sleep_start is None:
            print(f"  skipping {d}: no sleep window yet (run `sleeplab fetch` for it)")
            continue
        nights.append(n)
    idx = build_report(nights, a.out)
    print(f"Report: {idx.resolve()}")


def cmd_cover(a) -> None:
    from .cover import render_cover
    from .synthetic import PROFILES, generate_night

    render_cover(generate_night("2026-10-23", PROFILES[0], seed=7, tz=_tz()), a.out)
    print(f"Cover: {Path(a.out).resolve()}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    p = argparse.ArgumentParser(prog="sleeplab", description="Do I snore? Fitbit Air SpO2 meets overnight audio.")
    p.add_argument("--data", type=Path, default=Path("data"), help="data folder (default: data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="synthetic nights + report, no accounts needed")
    s.add_argument("--nights", type=int, default=3)
    s.add_argument("--last-date", default="2026-10-25")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", type=Path, default=Path("reports/demo"))
    s.set_defaults(func=cmd_demo, data_override=Path("data/demo"))

    s = sub.add_parser("auth", help="sign in with Google (browser) and store tokens")
    s.add_argument("--no-browser", action="store_true", help="print the URL instead of opening it")
    s.set_defaults(func=cmd_auth)

    s = sub.add_parser("whoami", help="check auth; show identity and paired devices")
    s.set_defaults(func=cmd_whoami)

    s = sub.add_parser("fetch", help="pull nights from the Google Health API")
    s.add_argument("--since", required=True, help="first wake-up date, YYYY-MM-DD")
    s.add_argument("--until", help="last wake-up date (default: today)")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("detect", help="find snores in an audio file")
    s.add_argument("file", type=Path)
    s.add_argument("--start", required=True, help="when the recording started, e.g. '2026-10-25 23:05' (local)")
    s.add_argument("--detector", choices=["yamnet", "energy"], default="yamnet")
    s.add_argument("--threshold", type=float, default=0.3)
    s.add_argument("--clips", type=int, default=10, help="keep N short clips of the loudest snores (0 = none)")
    s.set_defaults(func=cmd_detect)

    s = sub.add_parser("record", help="record tonight from the mic, keep only snore events")
    s.add_argument("--detector", choices=["yamnet", "energy"], default="yamnet")
    s.add_argument("--threshold", type=float, default=0.3)
    s.add_argument("--clips", type=int, default=20)
    s.add_argument("--keep-audio", type=Path, help="ALSO save the full night to this wav (off by default)")
    s.add_argument("--max-hours", type=float, default=11.0)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("report", help="build the HTML report from data/")
    s.add_argument("--out", type=Path, default=Path("reports"))
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("cover", help="render the showcase cover image")
    s.add_argument("--out", type=Path, default=Path("showcase/cover.png"))
    s.set_defaults(func=cmd_cover)

    a = p.parse_args(argv)
    if a.cmd == "demo" and a.data == Path("data"):
        a.data = a.data_override  # keep fake nights away from real ones
    try:
        a.func(a)
    except Exception as e:  # friendly errors for the expected failures
        from .google_health import AuthError

        if isinstance(e, AuthError):
            sys.exit(f"auth: {e}")
        raise


if __name__ == "__main__":
    main()
