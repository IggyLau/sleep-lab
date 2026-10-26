# Do I snore? (field report, draft)

*Status: the instrument is built and tested on synthetic nights. The real nights haven't happened yet. This draft gets rewritten once there's data from my own bedroom.*

> **Not a medical device.** A wrist SpO2 sensor and a pretrained audio classifier are not a sleep study. Nothing here diagnoses anything. If you're worried about your breathing at night, talk to a doctor, not a Python script.

## The question

People have told me I snore. I've never heard it myself. My Fitbit Air logs blood oxygen and sleep stages overnight, and my laptop has a microphone. So the plan was to record a night, find the snores and check whether my SpO2 does anything about them.

## The rig

- **Wrist:** the Fitbit Air reports sleep stages and SpO2. Fitbit's own docs warn that the SpO2 samples are 1-minute and "sparse", and gaps are expected.
- **Ears:** a laptop mic at 16 kHz, scored by **YAMNet**. It's Google's AudioSet classifier, and one of its 521 classes is literally *Snoring*, so no training was needed to start.
- **Glue:** everything goes on a 1-minute grid, and the output is one static HTML report per night.

## Field notes

**The API changed under me.** The spec said "use the Fitbit Web API". Checking first found that the Fitbit Web API has become the **Google Health API**. It now uses Google OAuth and new endpoints, and the old API reportedly switches off on 30 October 2026, four days after I started. Sleep Lab only talks to the new one. In Testing mode, Google expires the refresh token every 7 days, so signing in again is now a weekly chore.

**My laptop is not a data centre.** The first `pip install` failed with *no space left on device*. TensorFlow alone is about 1.5 GB. I cleared some space and moved on.

**Privacy shaped the design.** A full night of bedroom audio is not something I want lying around. So the recorder works in one-minute chunks: score the chunk, keep the timestamps of anything snore-shaped, throw the audio away. It keeps a handful of 4-second clips so I can check whether the "snore" was me or the radiator.

**Correlation needs a control.** SpO2 on a wrist wobbles a lot, so any snore bout will have some dip after it if you look hard enough. The report compares the average SpO2 change after real snore bouts with the same average at 300 sets of *random* sleeping minutes. A dip only counts if it falls outside that band.

## Results so far (synthetic nights only)

The pipeline is tested on generated nights with a known effect built in: snore bouts pull SpO2 down by about 1–2 points, starting about a minute later.

![Synthetic demo night](cover.png)

The analysis recovers the planted effect. Correlation is strongest at r ≈ −0.8 with a 3-minute lag, and the triggered average bottoms out around −2 points, outside the random band. The quiet night correctly reports "barely". That proves the instrument works, but it says nothing about me yet.

## Next

1. Record three or more real nights and replace this section with what my own data shows.
2. Listen to the kept clips and count how often YAMNet's "Snoring" really is snoring.
3. Add Garmin runs and a small MCP server so I can ask Claude how last night went.
