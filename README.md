# Sleep Lab

**Do I snore?** Sleep Lab lines up overnight blood oxygen (SpO2) and sleep stages from a **Fitbit Air** with snore events detected locally from an audio recording. It answers on one timeline: do SpO2 dips follow the snoring?

> **Not a medical device.** This is a curiosity project. Wrist SpO2 is 1-minute, gappy and motion-sensitive, and a pretrained audio classifier is not a sleep study. If you're worried about your breathing at night, see a doctor.

```
Fitbit Air ──Google Health API──► sleep stages, SpO2, heart rate, resting HR ─┐
                                                                              ├─► 1-minute grid ─► reports/index.html
laptop mic / audio file ──YAMNet "Snoring" (local)──► snore timestamps ───────┘
```

## Quick start (no accounts, no mic)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/sleeplab demo                 # 3 synthetic nights -> reports/demo/index.html
.venv/bin/python -m pytest -q
```

Open `reports/demo/index.html`. The synthetic nights contain a known effect (SpO2 dips about a minute after snore bouts), and the report should find it. One test checks that it does.

For real nights, also install the snore detector and recorder (about 1.5 GB, mostly TensorFlow):

```bash
.venv/bin/pip install -e '.[yamnet,record,dev]'
```

## API status (checked 26 Oct 2026)

Google has replaced the **Fitbit Web API** with the **Google Health API**. A Google developer checklist says the legacy API stops working on **30 Oct 2026**, so Sleep Lab targets only the new API.

| | |
|---|---|
| Base URL | `https://health.googleapis.com/v4` |
| Read data | `GET /users/me/dataTypes/{type}/dataPoints?filter=…&pageToken=…` |
| Types used | `sleep` (session with stages AWAKE/LIGHT/DEEP/REM), `oxygen-saturation` (1-min samples, "coverage can be sparse"), `heart-rate`, `daily-resting-heart-rate` |
| Auth | Google OAuth 2.0 (replaces Fitbit's own OAuth). All Google Health scopes are *restricted*. |
| Scopes requested | `googlehealth.sleep.readonly`, `googlehealth.health_metrics_and_measurements.readonly`, `googlehealth.activity_and_fitness.readonly` |
| Personal use | Fine in **Testing** mode with yourself as a test user, up to 100 users with no verification. **Refresh tokens expire after 7 days in Testing mode**, so expect to run `sleeplab auth` weekly. |
| Gotcha | You must first link your Google account in the Google Health mobile app, or `/users/me/identity` fails. |

Docs: [About](https://developers.google.com/health/about) · [Setup](https://developers.google.com/health/setup) · [Endpoints](https://developers.google.com/health/endpoints) · [Data types](https://developers.google.com/health/data-types) · [Filters](https://developers.google.com/health/filters) · [REST reference](https://developers.google.com/health/reference/rest)

Not yet confirmed against a real Fitbit Air: which scope covers SpO2 and heart rate (the docs don't say, so both candidate scopes are requested), and whether the list endpoint takes `pageSize` or `page_size`. The first `sleeplab fetch` saves the raw responses under `data/nights/<date>/raw/` so this is easy to check.

## Google Health API setup (one time, about 10 minutes)

You do all of this in your own browser and Google account. Sleep Lab never sees your password.

1. **Link the account.** In the Google Health app on your phone, sign in and link your Google account to your Fitbit Air.
2. **Create a project** at <https://console.cloud.google.com/projectcreate>, named something like `sleep-lab`.
3. **Enable the API** at <https://console.developers.google.com/apis/library/health.googleapis.com>: click **Enable** on *Google Health API*.
4. **Set up the consent screen** at <https://console.developers.google.com/auth/audience>:
   - Set User type to **External** and leave Publishing status as **Testing**.
   - Under **Test users**, click **+ Add users**, add your own Google email and save.
5. **Add scopes** at <https://console.developers.google.com/auth/scopes>:
   - Click **Add or remove scopes** and search "Google Health API".
   - Tick the three `.readonly` scopes listed above, then click **Update** and **Save**.
6. **Create the OAuth client** at <https://console.developers.google.com/apis/credentials>:
   - Choose **Create credentials → OAuth client ID → Web application**.
   - Under **Authorized redirect URIs**, add `http://localhost:8765/callback`.
   - Click **Create** and copy the Client ID and Client secret.
7. **Configure Sleep Lab**:
   ```bash
   cp .env.example .env      # then paste the client ID and secret into .env
   .venv/bin/sleeplab auth   # opens Google's consent page; tokens land in .tokens.json
   .venv/bin/sleeplab whoami # should print your healthUserId and paired devices
   ```

`.env` and `.tokens.json` are gitignored. `.tokens.json` is created with mode 600.

## A real night

```bash
# Bedtime: put the laptop near the bed, plugged in, lid open (or use caffeinate).
.venv/bin/sleeplab record                 # Ctrl-C when you wake up

# Or record with your phone and feed the file in afterwards (wav/flac/ogg/mp3; convert m4a with ffmpeg):
.venv/bin/sleeplab detect night.wav --start "2026-10-26 23:10"

# Morning, once the Fitbit has synced:
.venv/bin/sleeplab fetch --since 2026-10-27
.venv/bin/sleeplab report                 # -> reports/index.html
```

A night is filed under the local date you **woke up** on, which is Fitbit's convention. `SLEEPLAB_TZ` in `.env` sets the timezone.

To check snore detection on your own audio first, run `sleeplab detect clip.wav --start ... --clips 10` and listen to the clips it keeps. If YAMNet misses quiet snores, lower `--threshold` (default 0.3). `--detector energy` is a crude loudness fallback that needs no TensorFlow.

## Privacy

- **Processing is local.** YAMNet runs on your machine. The model is downloaded once from TF Hub, and audio is never uploaded.
- **`record` doesn't keep the night.** It works in 1-minute chunks: each chunk is scored, its snore timestamps are saved and the audio is discarded. It keeps up to `--clips` short clips (about 4 s each) so you can check them. The full night is saved only if you pass `--keep-audio FILE`.
- **`detect` never copies or changes your file.** It saves timestamps plus `--clips` short clips.
- **Health data stays local.** `data/`, `reports/`, audio files and tokens are all gitignored.

## What's in the report

- **Timeline:** sleep stage, SpO2 (▼ marks a 3-point dip), seconds of snoring per minute and heart rate on one time axis. Minutes with no audio are shaded, so "not listening" doesn't read as "not snoring".
- **Event-triggered average:** the average SpO2 change around each snore bout onset (3 or more snores at most 30 s apart), measured against the 5 minutes before it. It's compared with the same average at 300 sets of random sleeping minutes. If the dip stays inside the grey band, it's not distinguishable from chance.
- **Lagged cross-correlation:** r between seconds of snoring in minute *t* and SpO2 at *t + lag*. A dip that follows snoring shows up as negative r at small positive lags.
- **Index:** bouts per hour, dips per hour and a verdict for each night.

## Layout

```
src/sleeplab/
  google_health.py   OAuth (PKCE + loopback), API client, parsers
  ingest.py          fetch nights -> data/nights/<date>/ (+ raw/ API responses)
  audio.py           YAMNet + energy detectors, chunked file scoring, clips
  record.py          live mic recording, snores-only
  analysis.py        minute grid, bouts, cross-correlation, triggered average, verdict
  report.py          static HTML (plotly, offline)
  synthetic.py       fake nights and fake snore audio for demos and tests
  cover.py           showcase/cover.png
showcase/            what the personal website reads
```

## Roadmap

- [x] Phase 1: Google Health ingestion, local snore detection, one-timeline report
- [x] Phase 2 (partly): cross-correlation and event-triggered average with a shuffle baseline, multi-night index, privacy-first recording
- [ ] Run on 3 or more real nights (needs the setup above and some sleep)
- [ ] Garmin Forerunner 245 for runs, through a Garmin MCP server (Fitbit Air = sleep and resting HR; Garmin = runs)
- [ ] A small local MCP server over `data/` so Claude can answer "how was last night?"
- [ ] Stretch: fine-tune the snore detector on labelled clips; compare room temperature and humidity using `basecamp-node`
