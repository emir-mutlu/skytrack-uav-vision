"""Record a run of the operator console and build a static replay of it.

The output folder is the console UI driven by a recorded state stream (10 Hz) and an H.264
video of the annotated feed, so it can be hosted without a server, e.g. as a Hugging Face
Static Space or on GitHub Pages.

    pip install imageio-ffmpeg
    python scripts/export_replay.py --out site --seconds 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

import imageio_ffmpeg

from skytrack.config import load_config
from skytrack.console import ConsoleSession
from skytrack.simvehicle import SimVehicle
from skytrack.synthetic import PanningScene

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "skytrack" / "console" / "static"
FPS, STATE_HZ = 30, 10


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RecordingSession(ConsoleSession):
    """Console session that hands every published frame and its state to a callback, on the
    pipeline thread, so the recording cannot drop frames or pair a frame with another state."""

    def __init__(self, *args, sink, **kwargs):
        super().__init__(*args, **kwargs)
        self._sink = sink

    def _publish(self, jpeg: bytes | None) -> None:
        super()._publish(jpeg)
        if jpeg is not None:
            self._sink(jpeg, self._state_json)


def record(frames_dir: Path, seconds: float, warmup: float, seed: int) -> list[dict]:
    cfg = load_config(str(ROOT / "configs" / "demo.yaml"), {"output": {"show": False}})
    port = _free_port()
    scene = PanningScene(width=1280, height=720, hfov_deg=cfg.camera.hfov_deg, seed=seed, n_targets=3)
    pitch_lim, yaw_lim = scene.gimbal_limits()
    sim = SimVehicle(connection=f"tcpin:127.0.0.1:{port}", pitch_limits=pitch_lim, yaw_limits=yaw_lim).start()
    cfg.mavlink.enabled, cfg.mavlink.connection = True, f"tcp:127.0.0.1:{port}"
    cfg.mavlink.gimbal.follow = True
    frames = (scene.step(sim.gimbal_pitch, sim.gimbal_yaw)[0] for _ in itertools.count())

    first, n = int(warmup * FPS), int(seconds * FPS)
    states: dict[int, dict] = {}
    done = threading.Event()

    def sink(jpeg: bytes, text: str) -> None:
        state = json.loads(text)
        k = state.get("frame", -1) - first
        if 0 <= k < n:
            (frames_dir / f"{k:06d}.jpg").write_bytes(jpeg)
            if k % (FPS // STATE_HZ) == 0:
                states[k // (FPS // STATE_HZ)] = state
        elif k >= n:
            done.set()

    session = _RecordingSession(cfg, frames, FPS, False, "Simulation", read_only=True, sink=sink).start()
    try:
        done.wait(warmup + seconds + 120)
    finally:
        session.stop()
        sim.stop()
    got = sum(1 for _ in frames_dir.iterdir())
    print(f"recorded {got}/{n} frames, {len(states)}/{n * STATE_HZ // FPS} states")

    prev = None  # fill frames the recorder missed with the one before
    for k in range(n):
        f = frames_dir / f"{k:06d}.jpg"
        if f.exists():
            prev = f
        elif prev is not None:
            shutil.copyfile(prev, f)
    out, last = [], None
    for k in range(n * STATE_HZ // FPS):
        last = states.get(k, last)
        if last is not None:
            out.append(last)
    return out


def encode(frames_dir: Path, out: Path, crf: int) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames_dir / "%06d.jpg"), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)


def build_site(out: Path, states: list[dict], video: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("app.js", "app.css", "B612Mono-Regular.ttf", "B612Mono-Bold.ttf", "OFL-B612.txt"):
        shutil.copyfile(STATIC / name, out / name)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', '<html lang="en" data-replay>', 1)
    html = html.replace('<img id="video" alt="Payload video with tracking symbology">',
                        '<video id="video" src="replay.mp4" muted autoplay loop playsinline '
                        'aria-label="Recorded payload video with tracking symbology"></video>', 1)
    assert "data-replay" in html and "replay.mp4" in html, "console index.html changed; update export_replay.py"
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "states.json").write_text(json.dumps(states, separators=(",", ":")), encoding="utf-8")
    shutil.copyfile(video, out / "replay.mp4")
    shutil.copyfile(ROOT / "deploy" / "huggingface" / "README.md", out / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="site")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds to skip at the start")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crf", type=int, default=28, help="H.264 quality (lower is better, bigger)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir()
        print(f"recording {args.seconds:.0f} s of the simulation console ...")
        states = record(frames_dir, args.seconds, args.warmup, args.seed)
        video = Path(tmp) / "replay.mp4"
        encode(frames_dir, video, args.crf)
        out = Path(args.out)
        build_site(out, states, video)
    for f in sorted(out.iterdir()):
        print(f"{f}  {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
