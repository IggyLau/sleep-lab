# sleep-lab

Do I snore? This pulls SpO2 and sleep stages from my Fitbit Air, detects snoring from an overnight audio recording, and puts them on the same timeline to see if my oxygen drops after I snore.

Not a medical device. Just a side project.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

For snore detection and recording you also need TensorFlow and sounddevice (around 1.5 GB):

```bash
.venv/bin/pip install -e '.[yamnet,record,dev]'
```

## Try it without any data

```bash
.venv/bin/sleeplab demo
.venv/bin/python -m pytest -q
```

`demo` generates three fake nights and writes a report to `reports/demo/index.html`. The fake nights have a dip in SpO2 after each snore bout baked in, so you can check the analysis finds it.

## Fitbit / Google Health API

The Fitbit Web API has been replaced by the Google Health API, and the old one is supposed to stop working on 30 Oct 2026. This project only uses the new one.

- Base URL: `https://health.googleapis.com/v4`
- Data is read from `GET /users/me/dataTypes/{type}/dataPoints`
- Types used: `sleep`, `oxygen-saturation`, `heart-rate`, `daily-resting-heart-rate`
- Auth is normal Google OAuth. Scopes: `googlehealth.sleep.readonly`, `googlehealth.health_metrics_and_measurements.readonly`, `googlehealth.activity_and_fitness.readonly`
- With the app in Testing mode, refresh tokens expire after 7 days, so you have to run `sleeplab auth` again each week.

I haven't run this against a real account yet, so two things are unconfirmed: which scope SpO2/heart rate actually live under (I request both likely ones), and whether paging uses `pageSize` or `page_size`. `fetch` saves the raw responses to `data/nights/<date>/raw/` so it's easy to check.

Docs: https://developers.google.com/health

### Setup

1. In the Google Health app on your phone, link your Google account to the Fitbit. (Without this, API calls fail.)
2. Create a Google Cloud project: https://console.cloud.google.com/projectcreate
3. Enable the Google Health API: https://console.developers.google.com/apis/library/health.googleapis.com
4. On https://console.developers.google.com/auth/audience set user type to External, leave it in Testing, and add your own email as a test user.
5. On https://console.developers.google.com/auth/scopes add the three read-only scopes above.
6. On https://console.developers.google.com/apis/credentials create an OAuth client ID (Web application) with redirect URI `http://localhost:8765/callback`.
7. Then:

```bash
cp .env.example .env    # paste in the client ID and secret
.venv/bin/sleeplab auth
.venv/bin/sleeplab whoami
```

`.env` and `.tokens.json` are gitignored.

## Using it

```bash
# at night
.venv/bin/sleeplab record          # ctrl-c in the morning

# or with a recording from your phone (wav/flac/ogg/mp3)
.venv/bin/sleeplab detect night.wav --start "2026-10-26 23:10"

# in the morning, after the Fitbit syncs
.venv/bin/sleeplab fetch --since 2026-10-27
.venv/bin/sleeplab report          # reports/index.html
```

Nights are named by the date you wake up. Set your timezone with `SLEEPLAB_TZ` in `.env`.

If YAMNet is missing quiet snores, lower `--threshold` (default 0.3). `--detector energy` is a dumb loudness-based fallback if you don't want to install TensorFlow.

## Audio and privacy

Everything runs locally. `record` processes audio a minute at a time and only keeps the snore timestamps, plus up to 20 short clips so you can listen back. It doesn't save the full night unless you pass `--keep-audio`. `detect` doesn't touch the original file. `data/`, `reports/` and audio files are all gitignored.

## The report

Each night gets a page with:

- a timeline of sleep stage, SpO2, snoring (seconds per minute) and heart rate. Gaps in the audio are shaded.
- the average SpO2 change around the start of each snore bout, compared against random times in the night. If the line stays inside the grey band it's just noise.
- the correlation between snoring and SpO2 at different time lags.

The index page compares all nights.

## Code

```
src/sleeplab/
  google_health.py   auth, API client, parsing
  ingest.py          fetches nights into data/nights/<date>/
  audio.py           YAMNet + fallback detector
  record.py          mic recording
  analysis.py        stats
  report.py          HTML report
  synthetic.py       fake data for the demo and tests
  cover.py           showcase/cover.png
```

## Todo

- [ ] 3+ real nights
- [ ] Garmin Forerunner 245 run data (via a Garmin MCP server)
- [ ] MCP server over `data/`
- [ ] fine-tune the snore detector on my own clips
- [ ] compare with room temp/humidity from basecamp-node
