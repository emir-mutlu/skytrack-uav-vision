---
title: SkyTrack GCS
emoji: 🛩️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: UAV target detection, tracking and lock, live GCS demo
---

# SkyTrack GCS: live demo

Real-time target detection, multi-object tracking and target lock for UAV video, running here
in a closed-loop simulation: a simulated copter talks MAVLink to the pipeline, and the tracker
points the camera gimbal at the locked target.

What you are looking at:

- **Video:** payload video with tracker symbology (azimuth tape, elevation scale, track gates,
  `LOCK` tag, aircraft and target position).
- **Map** (picture-in-picture, press `M` to swap): north-up map with the sensor footprint,
  geolocated tracks and the vehicle track.
- **Instruments:** attitude indicator and heading indicator with the sensor field of view and
  the bearing to the target.
- **Target / Tracks / Messages:** lock status, boresight error, target coordinates and range,
  every confirmed track, lock and link messages.

This public copy is **read-only**, so visitors cannot change it for each other. Run it locally
to designate targets with a click, break the lock, freeze the video and switch the gimbal
between Follow and Hold:

```bash
pip install "skytrack[mavlink] @ git+https://github.com/emir-mutlu/skytrack-uav-vision"
skytrack console
```

Source, benchmark and documentation: https://github.com/emir-mutlu/skytrack-uav-vision
