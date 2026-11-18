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

What you are looking at:

- **Video:** payload video with tracker symbology (azimuth tape, elevation scale, track gates,
  `LOCK` tag).
- **Map** (picture-in-picture, press `M` to swap): north-up map with the sensor footprint,
  geolocated tracks and the vehicle track.
- **Instruments:** attitude indicator and heading indicator with the sensor field of view and
  the bearing to the target.
- **Target / Tracks / Messages:** lock status, boresight error, target coordinates and range,
  every confirmed track, lock and link messages.

The operator controls are off in the recording. Run it locally for the live console, where you
designate targets with a click, break the lock, freeze the video and switch the gimbal between
Follow and Hold:

```bash
pip install "skytrack[mavlink] @ git+https://github.com/emir-mutlu/skytrack-uav-vision"
skytrack console
```

Source, benchmark and documentation: https://github.com/emir-mutlu/skytrack-uav-vision
