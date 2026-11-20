"""Build the aerial-photo ground used by the synthetic scenes.

Downloads Dutch national aerial photography (Luchtfoto Actueel Ortho 8 cm RGB, Beeldmateriaal
Nederland, served by PDOK as open data under CC BY) for a rural area, mosaics the tiles,
resamples them to the scene's ground sample distance and writes `<out>.webp` plus a
`<out>.json` with the scale, the sun direction read off the photo's shadows, the windows the
scenes look at, and the attribution.

    python scripts/build_aerial.py --lat 52.1294 --lon 5.6229 --nx 14 --ny 10 \
        --sun -0.82 -0.58 --sun-elev 32 --out src/skytrack/data/vallei
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/EPSG:3857/{z:02d}/{x}/{y}.jpeg"
ATTRIBUTION = ("Aerial photo: Luchtfoto Actueel Ortho 8 cm RGB, Beeldmateriaal Nederland, "
               "via PDOK (open data, CC BY 4.0)")


def tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch(z: int, x: int, y: int, cache: Path) -> np.ndarray:
    f = cache / f"{z}_{x}_{y}.jpg"
    if not f.exists():
        headers = {"User-Agent": "skytrack-scene-builder/1.0"}
        req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            f.write_bytes(r.read())
        time.sleep(0.05)  # be gentle with the public service
    img = cv2.imdecode(np.frombuffer(f.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"tile {z}/{x}/{y} is not an image")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--nx", type=int, default=14, help="tiles across")
    ap.add_argument("--ny", type=int, default=10, help="tiles down")
    ap.add_argument("--gsd", type=float, default=0.2, help="metres per pixel of the output")
    ap.add_argument("--sun", type=float, nargs=2, required=True, metavar=("DX", "DY"),
                    help="image direction the shadows in the photo fall in")
    ap.add_argument("--sun-elev", type=float, required=True, help="sun elevation (deg) from shadow lengths")
    ap.add_argument("--fixed-view", type=int, nargs=2, default=None, metavar=("X", "Y"),
                    help="centre of the fixed-camera scene, output px")
    ap.add_argument("--panning-view", type=int, nargs=2, default=None, metavar=("X", "Y"),
                    help="centre of the pan/tilt scene, output px")
    ap.add_argument("--cache", default="data/aerial_tiles")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    cx, cy = tile_xy(args.lat, args.lon, args.zoom)
    x0, y0 = int(cx) - args.nx // 2, int(cy) - args.ny // 2
    rows = [np.hstack([fetch(args.zoom, x0 + i, y0 + j, cache) for i in range(args.nx)]) for j in range(args.ny)]
    mosaic = np.vstack(rows)
    ground_res = 156543.03392804097 * math.cos(math.radians(args.lat)) / 2**args.zoom
    scale = ground_res / args.gsd
    img = cv2.resize(mosaic, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".webp")), img, [cv2.IMWRITE_WEBP_QUALITY, args.quality])
    sun = np.array(args.sun, float)
    sun /= np.linalg.norm(sun)
    meta = {
        "gsd_m": args.gsd,
        "size": [w, h],
        "sun_dir": [round(float(v), 3) for v in sun],
        "sun_elev_deg": args.sun_elev,
        "views": {"fixed": args.fixed_view or [w // 2, h // 2], "panning": args.panning_view or [w // 2, h // 2]},
        "attribution": ATTRIBUTION,
        "source": {"service": "PDOK luchtfotorgb WMTS", "layer": "Actueel_orthoHR", "tilematrixset": "EPSG:3857",
                   "zoom": args.zoom, "tiles": [x0, y0, args.nx, args.ny], "ground_res_m": round(ground_res, 4)},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    size_mb = out.with_suffix(".webp").stat().st_size / 1e6
    print(f"{out.with_suffix('.webp')}: {w}x{h} px at {args.gsd} m/px, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
