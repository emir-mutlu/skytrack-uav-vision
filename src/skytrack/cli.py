"""Command-line interface.

    skytrack run   --source video.mp4 --config configs/recon.yaml
    skytrack run   --source rtsp://192.168.1.10:8554/cam --detector yolo --model best.onnx
    skytrack demo  --save runs/demo.mp4
    skytrack eval  --frames 600
    skytrack synth --out data/synthetic.mp4
    skytrack sim-vehicle                                   # MAVLink test vehicle on tcp:5760
    skytrack demo --mavlink tcp:127.0.0.1:5760 --gimbal-follow
    skytrack link-check --mavlink COM5 --gimbal-test         # hardware bring-up
    skytrack console                                         # web operator console (simulation)
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
    if getattr(args, "mavlink", None):
        o["mavlink"] = {"enabled": True, "connection": args.mavlink}
    if getattr(args, "gimbal_follow", False):
        o.setdefault("mavlink", {})["gimbal"] = {"follow": True}
    if getattr(args, "hfov", None):
        o["camera"] = {"hfov_deg": args.hfov}
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
    p.add_argument("--mavlink", metavar="CONN",
                   help="connect to the flight controller: tcp:127.0.0.1:5760 (SITL), udpin:0.0.0.0:14550, COM5")
    p.add_argument("--gimbal-follow", action="store_true",
                   help="command gimbal rates to keep the designated target centred")
    p.add_argument("--hfov", type=float, help="camera horizontal field of view in degrees")


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline
    from .sources import _is_live, open_source

    cfg = load_config(args.config, _overrides(args))
    frames, fps = open_source(args.source, args.max_frames)
    summary = Pipeline(cfg).run(frames, fps, live=_is_live(args.source))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.closed_loop:
        return _closed_loop_demo(args)
    args.source = f"synthetic:{args.seed}"
    if not args.config:
        args.config = str(Path(__file__).resolve().parents[2] / "configs" / "demo.yaml")
        if not Path(args.config).exists():
            args.config = None
    if args.max_frames is None:
        args.max_frames = 600
    return cmd_run(args)


def _closed_loop_demo(args: argparse.Namespace) -> int:
    """Synthetic camera on a simulated MAVLink gimbal: the full detect -> lock -> gimbal
    command -> camera motion loop, talking real MAVLink over TCP."""
    from .pipeline import Pipeline
    from .simvehicle import SimVehicle
    from .synthetic import PanningScene

    port = args.sim_port
    args.mavlink = f"tcp:127.0.0.1:{port}"
    args.gimbal_follow = True
    if not args.config:
        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
        args.config = str(cfg_path) if cfg_path.exists() else None
    cfg = load_config(args.config, _overrides(args))
    scene = PanningScene(width=1280, height=720, hfov_deg=cfg.camera.hfov_deg, seed=args.seed, n_targets=3)
    pitch_lim, yaw_lim = scene.gimbal_limits()
    sim = SimVehicle(connection=f"tcpin:127.0.0.1:{port}", pitch_limits=pitch_lim, yaw_limits=yaw_lim).start()
    n = args.max_frames or 900
    frames = (scene.step(sim.gimbal_pitch, sim.gimbal_yaw)[0] for _ in range(n))
    try:
        summary = Pipeline(cfg).run(frames, 30.0, live=True)
    finally:
        sim.stop()
    summary["gimbal_commands"] = sim.gimbal_commands
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0


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


def cmd_sim_vehicle(args: argparse.Namespace) -> int:
    import time

    from .simvehicle import SimVehicle

    sim = SimVehicle(connection=args.connection, altitude=args.altitude, radius=args.radius,
                     speed=args.speed).start()
    print(f"simulated vehicle serving MAVLink on {args.connection}  (Ctrl+C to stop)")
    t_end = time.monotonic() + args.duration if args.duration else None
    try:
        while t_end is None or time.monotonic() < t_end:
            time.sleep(2.0)
            print(f"gimbal pitch {sim.gimbal_pitch:+6.1f}  yaw {sim.gimbal_yaw:+6.1f}  "
                  f"rate cmds {sim.gimbal_commands}")
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
    return 0


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def cmd_console(args: argparse.Namespace) -> int:
    import itertools
    import webbrowser

    from .console import ConsoleServer, ConsoleSession
    from .sources import _is_live, open_source

    src = args.source
    if not args.config and (src == "sim" or src.startswith("synthetic")):
        demo = Path(__file__).resolve().parents[2] / "configs" / "demo.yaml"
        args.config = str(demo) if demo.exists() else None
    cfg = load_config(args.config, _overrides(args))
    cfg.output.show = False
    sim = None
    if src == "sim":
        from .simvehicle import SimVehicle
        from .synthetic import PanningScene

        port = _free_port()
        scene = PanningScene(width=1280, height=720, hfov_deg=cfg.camera.hfov_deg, seed=args.seed, n_targets=3)
        pitch_lim, yaw_lim = scene.gimbal_limits()
        sim = SimVehicle(connection=f"tcpin:127.0.0.1:{port}", pitch_limits=pitch_lim, yaw_limits=yaw_lim).start()
        cfg.mavlink.enabled, cfg.mavlink.connection = True, f"tcp:127.0.0.1:{port}"
        cfg.mavlink.gimbal.follow = True
        frames = (scene.step(sim.gimbal_pitch, sim.gimbal_yaw)[0] for _ in itertools.count())
        fps, live, label = 30.0, False, "Simulation"
    else:
        frames, fps = open_source(src, args.max_frames or (10**9 if src.startswith("synthetic") else None))
        live = _is_live(src)
        label = "Synthetic" if src.startswith("synthetic") else ("Live" if live else Path(src).name)

    session = ConsoleSession(cfg, frames, fps, live, label).start()
    try:
        server = ConsoleServer(session, args.host, args.port)
    except OSError as exc:
        session.stop()
        if sim:
            sim.stop()
        print(f"error: cannot listen on {args.host}:{args.port} ({exc}); try --port {args.port + 1}")
        return 1
    print(f"SkyTrack console running at {server.url}  (Ctrl+C to stop)")
    if args.host not in ("127.0.0.1", "localhost"):
        print("warning: the console is reachable from the network; anyone who can open it can steer the gimbal")
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        session.stop()
        if sim:
            sim.stop()
    return 0


def cmd_link_check(args: argparse.Namespace) -> int:
    from .linkcheck import run_link_check

    cfg = load_config(args.config, {"mavlink": {"enabled": True, "connection": args.mavlink,
                                                **({"baud": args.baud} if args.baud else {})}}).mavlink
    ok = run_link_check(cfg, window_s=args.window, timeout_s=args.timeout, gimbal_test=args.gimbal_test)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="skytrack", description="UAV target detection, tracking & lock")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="process a video file, camera index or stream URL")
    p.add_argument("--source", "-s", required=True)
    _common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="run on the built-in synthetic scene")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--closed-loop", action="store_true",
                   help="camera on a simulated MAVLink gimbal that follows the locked target")
    p.add_argument("--sim-port", type=int, default=5770, help="TCP port for the simulated vehicle")
    _common(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("eval", help="benchmark detector + tracker on synthetic ground truth")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    _common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("sim-vehicle", help="run a minimal MAVLink vehicle (telemetry + gimbal) for testing")
    p.add_argument("--connection", default="tcpin:127.0.0.1:5760")
    p.add_argument("--altitude", type=float, default=100.0)
    p.add_argument("--radius", type=float, default=150.0)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--duration", type=float, help="stop after N seconds (default: run until Ctrl+C)")
    p.set_defaults(func=cmd_sim_vehicle)

    p = sub.add_parser("console", help="web operator console: live video, tactical plot, tracks, lock control")
    p.add_argument("--source", "-s", default="sim",
                   help="'sim' (default): closed-loop simulation with a MAVLink vehicle; 'synthetic', a video "
                        "file, camera index or stream URL")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--host", default="127.0.0.1", help="interface to serve on (default: this computer only)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    _common(p)
    p.set_defaults(func=cmd_console)

    p = sub.add_parser("link-check", help="diagnose the flight-controller link before a flight")
    p.add_argument("--mavlink", default="udpin:0.0.0.0:14550", metavar="CONN")
    p.add_argument("--baud", type=int)
    p.add_argument("--config", "-c")
    p.add_argument("--window", type=float, default=5.0, help="seconds to measure stream rates")
    p.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for a heartbeat")
    p.add_argument("--gimbal-test", action="store_true", help="sweep the gimbal yaw and verify it moved")
    p.set_defaults(func=cmd_link_check)

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
