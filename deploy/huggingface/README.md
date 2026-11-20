---
title: SkyTrack GCS
emoji: 🛩️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: UAV target detection, tracking and lock (GCS demo)
---

# SkyTrack GCS: recorded demo

Real-time target detection, multi-object tracking and target lock for UAV video. This page
replays a one-minute run of the SkyTrack ground-control console in a closed-loop simulation:
a simulated copter talks MAVLink to the pipeline, and the tracker points the camera gimbal at
the locked target.

What you are looking at, laid out like glass-cockpit displays:

- **SENS:** payload video with tracker symbology (azimuth tape, elevation scale, track gates,
  `LOCK` tag) over a real aerial photograph of Dutch farmland, with synthetic target boards.
- **PFD:** flight modes, attitude, speed and altitude tapes, heading with the sensor (cyan) and
  target (magenta) bugs.
- **HSD:** range rings, sensor footprint, line of sight and tracks as MIL-STD-2525 frames;
  `SWAP` (or `M`) swaps it with the video.
- **TGT DATA / TRACK FILE / CAS / MSG:** target grid reference (MGRS), position, ranges and
  lock time, every confirmed track, and dark-cockpit cautions with the message log.

The operator controls are off in the recording. Run it locally for the live console, where you
designate targets with a click, break the lock, freeze the video and switch the gimbal between
Follow and Hold:

```bash
pip install "skytrack[mavlink] @ git+https://github.com/emir-mutlu/skytrack-uav-vision"
skytrack console
```

Source, benchmark and documentation: https://github.com/emir-mutlu/skytrack-uav-vision

Aerial photo: Luchtfoto Actueel Ortho 8 cm RGB, Beeldmateriaal Nederland, via PDOK (open data,
CC BY 4.0). Cockpit font: B612 Mono (SIL Open Font License 1.1).
