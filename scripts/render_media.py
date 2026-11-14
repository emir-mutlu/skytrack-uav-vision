"""Render the README media (HUD screenshot + animated demo) from the synthetic scene.

    python scripts/render_media.py                 # -> assets/hud.png, assets/demo.webp

Everything is deterministic (fixed seed), so the media can be regenerated after any
HUD or pipeline change and always matches the code in the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from skytrack.config import load_config
from skytrack.hud import draw_hud
from skytrack.pipeline import Pipeline
from skytrack.synthetic import SyntheticScene

ROOT = Path(__file__).resolve().parents[1]


def render(width: int, height: int, frames: int, seed: int, noise: float):
    cfg = load_config(ROOT / "configs" / "demo.yaml", {"output": {"show": False}})
    pipe = Pipeline(cfg)
    scene = SyntheticScene(width=width, height=height, seed=seed, noise=noise)
    for i, (frame, _) in enumerate(scene.frames(frames)):
        res = pipe.process(frame, i, i / 30.0)
        yield i, res, draw_hud(frame.copy(), res.tracks, res.lock, pipe.locker, 30.0, res.latency_ms,
                               pipe.detector.name, profile=cfg.profile, frame_idx=i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=int, default=72)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) HD screenshot: first frame after the lock is declared + 1 s
    shot = None
    for _, res, img in render(1280, 720, 400, args.seed, noise=2.0):
        if res.lock.state.value == "LOCKED" and res.lock.lock_duration >= 1.0:
            shot = img
            break
    if shot is None:
        raise SystemExit("no lock acquired in the screenshot window")
    cv2.imwrite(str(out / "hud.png"), shot, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # 2) animated demo (search -> acquire -> lock) as animated WebP: ~10x smaller than GIF
    #    at better quality, and rendered inline by GitHub READMEs and all modern browsers.
    w = args.width
    h = w * 9 // 16
    step = max(1, round(30 / args.fps))
    frames = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
              for i, _, img in render(w, h, int(args.seconds * 30), args.seed, noise=0.0) if i % step == 0]
    frames[0].save(out / "demo.webp", save_all=True, append_images=frames[1:], duration=int(1000 * step / 30),
                   loop=0, quality=args.quality, method=6)
    for f in ("hud.png", "demo.webp"):
        print(f"{out / f}  {(out / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
