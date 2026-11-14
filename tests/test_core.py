import json

import numpy as np
import pytest

from skytrack.config import Config, LockConfig, TrackerConfig, load_config
from skytrack.detectors import ClassicTargetDetector
from skytrack.evaluate import evaluate
from skytrack.lock import LockState, TargetLock
from skytrack.pipeline import Pipeline
from skytrack.synthetic import SyntheticScene
from skytrack.tracking import MultiObjectTracker
from skytrack.types import Detection, Track, iou, iou_matrix

SHAPE = (540, 960, 3)


# ---------------------------------------------------------------- geometry
def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_matrix_matches_scalar():
    a = [(0, 0, 10, 10), (5, 5, 20, 20)]
    b = [(0, 0, 10, 10), (5, 0, 15, 10), (100, 100, 110, 110)]
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            assert m[i, j] == pytest.approx(iou(x, y), abs=1e-5)


# ---------------------------------------------------------------- config
def test_config_overrides_and_validation():
    cfg = load_config(overrides={"detector": {"conf": 0.5}, "lock": {"zone": [0.1, 0.1, 0.9, 0.9]}})
    assert cfg.detector.conf == 0.5
    assert cfg.lock.zone == (0.1, 0.1, 0.9, 0.9)
    with pytest.raises(KeyError):
        load_config(overrides={"detector": {"nope": 1}})
    with pytest.raises(ValueError):
        load_config(overrides={"lock": {"zone": [0.9, 0.1, 0.1, 0.9]}})


@pytest.mark.parametrize("name", ["demo", "recon", "sar", "counter_uas"])
def test_bundled_profiles_load(name):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.profile == name


# ---------------------------------------------------------------- detector
def test_classic_detector_finds_targets_and_rejects_distractors():
    scene = SyntheticScene(seed=3, noise=3, occlusion=False)
    det = ClassicTargetDetector()
    hits = misses = fps = 0
    for frame, gt in scene.frames(30):
        dets = det.detect(frame)
        m = iou_matrix(gt, [d.bbox for d in dets])
        matched = int((m.max(axis=1) >= 0.5).sum()) if m.size else 0
        hits += matched
        misses += len(gt) - matched
        fps += len(dets) - matched
    assert hits / (hits + misses) > 0.9
    assert fps / max(hits, 1) < 0.1


def test_classic_detector_empty_frame():
    assert ClassicTargetDetector().detect(np.zeros(SHAPE, np.uint8)) == []


# ---------------------------------------------------------------- tracker
def test_tracker_keeps_identity_through_motion_and_dropout():
    trk = MultiObjectTracker(TrackerConfig(min_hits=2, max_age=5))
    ids = set()
    for i in range(20):
        if i in (10, 11):  # two missed detections
            tracks = trk.update([])
        else:
            x = 100 + 5 * i
            tracks = trk.update([Detection((x, 100, x + 40, 140), 0.9)])
        ids |= {t.track_id for t in tracks}
    assert ids == {1}
    assert tracks[0].velocity[0] == pytest.approx(5, abs=1.5)


def test_tracker_separates_two_objects_and_expires():
    trk = MultiObjectTracker(TrackerConfig(min_hits=1, max_age=2))
    for i in range(5):
        tracks = trk.update([Detection((10 + i, 10, 50 + i, 50), 0.9),
                             Detection((300, 300 + i, 340, 340 + i), 0.9)])
    assert len({t.track_id for t in tracks}) == 2
    for _ in range(4):
        tracks = trk.update([])
    assert tracks == []


# ---------------------------------------------------------------- lock
def _track(tid=1, cx=480, cy=270, s=60, confirmed=True, label="target"):
    return Track(tid, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), 0.9, 0, label,
                 confirmed=confirmed)


def test_auto_lock_sequence_and_loss():
    lk = TargetLock(LockConfig(acquire_seconds=1.0, lost_grace_seconds=0.2,
                               require_operator_confirm=False))
    assert lk.update([_track()], SHAPE, 0.0).state == LockState.ACQUIRING
    assert lk.update([_track()], SHAPE, 0.5).progress == pytest.approx(0.5)
    st = lk.update([_track()], SHAPE, 1.0)
    assert st.state == LockState.LOCKED
    assert [e.kind for e in st.events] == ["lock_acquired"]
    # short dropout is tolerated
    assert lk.update([], SHAPE, 1.1).state == LockState.LOCKED
    st = lk.update([], SHAPE, 1.5)
    assert st.state == LockState.LOST
    assert st.events[0].kind == "lock_lost" and st.events[0].duration == pytest.approx(0.5)


def test_lock_requires_target_inside_zone_and_big_enough():
    lk = TargetLock(LockConfig(require_operator_confirm=False, min_size_frac=0.05))
    assert lk.update([_track(cx=40)], SHAPE, 0).state == LockState.SEARCHING  # outside zone
    assert lk.update([_track(s=10)], SHAPE, 0).state == LockState.SEARCHING  # too small
    assert lk.update([_track(confirmed=False)], SHAPE, 0).state == LockState.SEARCHING


def test_label_filter():
    lk = TargetLock(LockConfig(require_operator_confirm=False, target_labels=["drone"]))
    assert lk.update([_track(label="bird")], SHAPE, 0).state == LockState.SEARCHING
    assert lk.update([_track(label="drone")], SHAPE, 0).state == LockState.ACQUIRING


def test_operator_in_the_loop():
    lk = TargetLock(LockConfig(acquire_seconds=0.5, require_operator_confirm=True))
    tracks = [_track(1, cx=420), _track(2, cx=560)]
    st = lk.update(tracks, SHAPE, 0)
    assert st.state == LockState.SEARCHING and st.candidate_id in (1, 2)
    lk.designate(2, 0.1)
    lk.update(tracks, SHAPE, 0.1)
    st = lk.update(tracks, SHAPE, 0.7)
    assert st.state == LockState.LOCKED and st.track_id == 2
    lk.release(0.8)
    assert lk.update(tracks, SHAPE, 0.9).state == LockState.SEARCHING


def test_centre_error_normalised():
    err = TargetLock.centre_error(_track(cx=960, cy=0), SHAPE)
    assert err == pytest.approx((1.0, -1.0))


# ---------------------------------------------------------------- end to end
def test_pipeline_end_to_end(tmp_path):
    cfg = load_config("configs/demo.yaml", {"output": {
        "show": False, "telemetry": str(tmp_path / "t.jsonl"), "save_video": str(tmp_path / "o.mp4")}})
    scene = SyntheticScene(seed=7)
    summary = Pipeline(cfg).run((f for f, _ in scene.frames(150)), 30.0)
    assert summary["frames"] == 150
    assert summary["locks_acquired"] >= 1
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    assert len(lines) == 150
    rec = json.loads(lines[-1])
    assert {"frame", "tracks", "lock", "latency_ms"} <= rec.keys()
    assert (tmp_path / "o.mp4").stat().st_size > 0


def test_benchmark_regression_gate():
    """Guards against accuracy regressions in detector / tracker changes."""
    rep = evaluate(load_config("configs/demo.yaml"), frames=200)
    assert rep.det_f1 > 0.9
    assert rep.mota > 0.85
    assert rep.id_switches <= 3


def test_default_config_roundtrip():
    assert Config().to_dict()["detector"]["kind"] == "classic"


# ---------------------------------------------------------------- HUD
@pytest.mark.parametrize("state", list(LockState))
@pytest.mark.parametrize("shape", [(480, 640, 3), (1080, 1920, 3)])
def test_hud_renders_every_state_at_any_resolution(state, shape):
    from skytrack.hud import draw_hud
    from skytrack.lock import LockStatus

    h, w = shape[:2]
    frame = np.full(shape, 90, np.uint8)
    tr = _track(cx=w // 2 + 40, cy=h // 2, s=h // 10)
    tr.history = [(w // 2, h // 2 + i) for i in range(10)]
    status = LockStatus(state=state, track_id=1, bbox=tr.bbox, progress=0.4, error=(0.1, -0.2),
                        lock_duration=1.5)
    out = draw_hud(frame.copy(), [tr], status, TargetLock(), 30.0, 5.0, "classic",
                   paused=True, profile="demo", frame_idx=12)
    assert out.shape == shape and out.dtype == np.uint8
    assert (out != frame).any()


def test_hud_track_only_mode_and_fonts():
    from skytrack.hud import _sprite, draw_hud, get_font
    from skytrack.lock import LockStatus

    for kind in ("sans", "bold", "mono"):
        assert get_font(kind, 14) is not None
    fill, _, width = _sprite("12.3 ms", "mono", 14, 0)
    assert width > 0 and fill.max() > 0.5
    frame = np.zeros((540, 960, 3), np.uint8)
    out = draw_hud(frame, [], LockStatus(), TargetLock(LockConfig(enabled=False)), 0.0, 0.0, "yolo")
    assert out.shape == (540, 960, 3)
