"""Command-line interface.

    skytrack run   --source video.mp4 --config configs/recon.yaml
    skytrack run   --source rtsp://192.168.1.10:8554/cam --detector yolo --model best.onnx
    skytrack demo  --save runs/demo.mp4
    skytrack eval  --frames 600
    skytrack synth --out data/synthetic.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .config import load_config


def _overrides(args: argparse.Namespace) -> dict:
    o: dict = {"detector": {}, "output": {}, "lock": {}}
    if getattr(args, "detector", None):
        o["detector"]["kind"] = args.detector
    if getattr(args, "model", None):
        o["detector"]["model"] = args.model
    if getattr(args, "conf", None) is not None:
        o["detector"]["conf"] = args.conf
    if getattr(args, "device", None):
        o["detector"]["device"] = args.device
    if getattr(args, "save", None):
        o["output"]["save_video"] = args.save
    if getattr(args, "telemetry", None):
        o["output"]["telemetry"] = args.telemetry
    if getattr(args, "no_show", False):
        o["output"]["show"] = False
    if getattr(args, "auto_lock", False):
        o["lock"]["require_operator_confirm"] = False
    return {k: v for k, v in o.items() if v}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="YAML mission profile (configs/*.yaml)")
    p.add_argument("--detector", choices=["classic", "yolo"])
    p.add_argument("--model", help="YOLO weights (.pt / .onnx / .engine)")
    p.add_argument("--conf", type=float)
    p.add_argument("--device", help="cpu | 0 | mps")
    p.add_argument("--save", help="write annotated video to this path (.mp4)")
    p.add_argument("--telemetry", help="write per-frame JSONL telemetry to this path")
    p.add_argument("--no-show", action="store_true", help="headless mode (no window)")
    p.add_argument("--auto-lock", action="store_true",
                   help="auto-select the lock candidate instead of waiting for operator designation")
    p.add_argument("--max-frames", type=int)


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline
    from .sources import _is_live, open_source

    cfg = load_config(args.config, _overrides(args))
    frames, fps = open_source(args.source, args.max_frames)
    summary = Pipeline(cfg).run(frames, fps, live=_is_live(args.source))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    args.source = f"synthetic:{args.seed}"
    if not args.config:
        args.config = str(Path(__file__).resolve().parents[2] / "configs" / "demo.yaml")
        if not Path(args.config).exists():
            args.config = None
    if args.max_frames is None:
        args.max_frames = 600
    return cmd_run(args)


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate

    cfg = load_config(args.config, _overrides(args))
    report = evaluate(cfg, frames=args.frames, seed=args.seed)
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from .synthetic import SyntheticScene

    scene = SyntheticScene(seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (scene.w, scene.h))
    for frame, _ in scene.frames(args.frames):
        vw.write(frame)
    vw.release()
    print(f"wrote {args.frames} frames -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="skytrack", description="UAV target detection, tracking & lock")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="process a video file, camera index or stream URL")
    p.add_argument("--source", "-s", required=True)
    _common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="run on the built-in synthetic scene")
    p.add_argument("--seed", type=int, default=7)
    _common(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("eval", help="benchmark detector + tracker on synthetic ground truth")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    _common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("synth", help="export the synthetic scene as a video file")
    p.add_argument("--out", default="data/synthetic.mp4")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_synth)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
