"""Draw the ROI by hand, and see what it costs while you draw it.

    python -m diffusion_policy.drim.roi_tool --data <zarr>

The measured boxes in :mod:`~diffusion_policy.drim.dressing` are placed by
motion energy alone, which knows where things move and nothing about whether
what is left is enough to act on. This puts that judgement back with a person,
while still reporting the three numbers that decide whether a box is a good
one:

``energy``    share of the total per-pixel temporal motion energy inside the
              box. How much of the action you kept.
``px/out``    native pixels per output pixel after the resize to the encoder's
              input. Below 1.0 the frames are upsampled and the extra pixels
              are interpolation, not detail; 1.00 exactly is the sweet spot.
``d'``        between-episode over within-episode brightness spread inside the
              box. How separable the episodes are **by illumination alone** —
              a cue a policy can key on to identify which demonstration it is
              in, which is a shortcut to the trajectory that does not survive a
              different hour. Lower is better.

Controls
--------
``drag``            draw a box
``arrows``          nudge the box (with ``shift``: resize)
``[`` / ``]``       scrub frames        ``,`` / ``.``  previous / next episode
``h``               heat-map overlay (off by default)
``p``               preview what the encoder sees
``c``               switch camera       ``r``  clear the box
``1`` ``2`` ``3``   load the wide / mid / hand preset
``s``               save to --out       ``q`` / ``esc``  quit
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: cv2 reports arrow keys differently depending on the GUI backend.
_LEFT, _RIGHT, _UP, _DOWN = (81, 2, 65361), (83, 3, 65363), (82, 0, 65362), (84, 1, 65364)
_ARROWS = set(_LEFT + _RIGHT + _UP + _DOWN)

#: Frames sampled for the statistics.  Enough for a stable brightness estimate
#: per episode without holding the whole recording in memory.
N_SAMPLE = 240


def _load(zarr_path: str, cameras: List[str]):
    import zarr

    g = zarr.open(zarr_path, "r")
    d = g["data"]
    ends = np.asarray(g["meta/episode_ends"][:], np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    n = int(ends[-1])
    idx = np.unique(np.linspace(0, n - 1, min(N_SAMPLE, n)).astype(np.int64))
    ep_of = np.zeros(n, np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        ep_of[a:b] = i
    out = {}
    for cam in cameras:
        sample = d[cam].oindex[idx].astype(np.float32).mean(-1)      # [S,H,W]
        energy = np.abs(np.diff(sample, axis=0)).mean(0)
        out[cam] = {"sample": sample, "energy": energy,
                    "ep": ep_of[idx], "arr": d[cam]}
    return out, list(zip(starts, ends))


def _box_stats(c: Dict[str, Any], box: Tuple[int, int, int, int],
               image_size: Tuple[int, int]) -> Dict[str, float]:
    y0, x0, h, w = box
    if h < 8 or w < 8:
        return {}
    e = c["energy"]
    share = float(e[y0:y0 + h, x0:x0 + w].sum() / max(e.sum(), 1e-9))
    means = c["sample"][:, y0:y0 + h, x0:x0 + w].reshape(len(c["sample"]), -1).mean(1)
    eps = np.unique(c["ep"])
    per = [means[c["ep"] == k] for k in eps]
    within = float(np.mean([p.std() for p in per if len(p) > 1]) or 0.0)
    between = float(np.std([p.mean() for p in per]))
    return {"energy": share,
            "px_per_out": (h * w) / float(image_size[0] * image_size[1]),
            "brightness": float(means.mean()),
            "d_prime": between / max(within, 1e-9),
            "within": within, "between": between}


def run(zarr_path: str, cameras: List[str], out_path: str,
        image_size: Tuple[int, int]) -> Dict[str, Tuple[int, int, int, int]]:
    import cv2

    from diffusion_policy.drim import dressing as D

    data, episodes = _load(zarr_path, cameras)
    ci = 0
    cam = cameras[ci]
    H, W = data[cam]["energy"].shape
    frame_i, ep_i = 0, 0
    show_heat, show_prev = False, True
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    drag: Dict[str, Any] = {"on": False, "p0": None, "p1": None}
    WIN = "roi  (drag=draw  arrows=nudge  hpc123 rs q)"

    def cur_box() -> Optional[Tuple[int, int, int, int]]:
        """The box being drawn, else the stored one.

        Never returns a degenerate box: a click without a drag has ``p0 == p1``,
        which is a zero-size crop, and ``cv2.resize`` raises on an empty image
        rather than returning one.
        """
        if drag["p0"] and drag["p1"]:
            (ax, ay), (bx, by) = drag["p0"], drag["p1"]
            y0, y1 = sorted((max(0, min(ay, H - 1)), max(0, min(by, H - 1))))
            x0, x1 = sorted((max(0, min(ax, W - 1)), max(0, min(bx, W - 1))))
            if y1 - y0 >= 2 and x1 - x0 >= 2:
                return (y0, x0, y1 - y0, x1 - x0)
        return boxes.get(cam)

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(on=True, p0=(x, y), p1=(x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drag["on"]:
            drag["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["on"]:
            drag.update(on=False, p1=(x, y))
            b = cur_box()
            if b and b[2] >= 8 and b[3] >= 8:
                boxes[cam] = b
            drag.update(p0=None, p1=None)

    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    except cv2.error as exc:                                # pragma: no cover
        raise SystemExit(f"no display available for the ROI tool: {exc}")
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cam = cameras[ci]
        a, b = episodes[ep_i]
        t = int(np.clip(a + frame_i, a, b - 1))
        base = cv2.cvtColor(np.asarray(data[cam]["arr"][t]), cv2.COLOR_RGB2BGR)
        view = base.copy()
        if show_heat:
            e = data[cam]["energy"]
            hm = cv2.applyColorMap(((e / e.max()) * 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)
            view = cv2.addWeighted(view, 0.6, hm, 0.4, 0)

        box = cur_box()
        lines = [f"{cam}   ep {ep_i + 1}/{len(episodes)}   frame {t - a}/{b - a}"]
        if box:
            y0, x0, h, w = box
            cv2.rectangle(view, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
            st = _box_stats(data[cam], box, image_size)
            lines.append(f"(y0={y0}, x0={x0}, h={h}, w={w})")
            if st:
                lines.append(f"energy {st['energy']:.1%}   px/out {st['px_per_out']:.2f}"
                             f"   d' {st['d_prime']:.1f}   brightness {st['brightness']:.0f}")
                lines.append("px/out 1.00 = no resampling      d' lower = harder to "
                             "tell episodes apart by light")
        else:
            lines.append("drag to draw a box;  1/2/3 loads wide/mid/hand")

        panel = np.full((26 * len(lines) + 12, view.shape[1], 3), 24, np.uint8)
        for i, s in enumerate(lines):
            cv2.putText(panel, s, (10, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1)
        stack = [view, panel]
        if show_prev and box and box[2] >= 2 and box[3] >= 2:
            y0, x0, h, w = box
            crop = cv2.resize(base[y0:y0 + h, x0:x0 + w],
                              (image_size[1], image_size[0]),
                              interpolation=cv2.INTER_AREA)
            pad = np.full((image_size[0], view.shape[1], 3), 24, np.uint8)
            pad[:, :image_size[1]] = crop
            cv2.putText(pad, "what the encoder sees",
                        (image_size[1] + 14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            stack.append(pad)
        cv2.imshow(WIN, np.vstack(stack))

        k = cv2.waitKey(20) & 0xFFFF
        if k in (ord("q"), 27):
            break
        elif k == ord("h"):
            show_heat = not show_heat
        elif k == ord("p"):
            show_prev = not show_prev
        elif k == ord("c"):
            ci = (ci + 1) % len(cameras)
        elif k == ord("r"):
            boxes.pop(cam, None)
        elif k in (ord("["), ord("]")):
            frame_i = max(0, frame_i + (8 if k == ord("]") else -8))
        elif k in (ord(","), ord(".")):
            ep_i = (ep_i + (1 if k == ord(".") else -1)) % len(episodes)
            frame_i = 0
        elif k in (ord("1"), ord("2"), ord("3")):
            preset = {ord("1"): D.ROI_WIDE, ord("2"): D.ROI_MID,
                      ord("3"): D.ROI_HAND}[k]
            if cam in preset:
                boxes[cam] = preset[cam]
        elif k == ord("s"):
            _save(boxes, out_path, data, image_size)
        elif box is not None and k in _ARROWS:
            #: arrow keys differ between cv2 builds, so both encodings are
            #: accepted; the guard is on ``box`` because there is nothing to
            #: nudge before one is drawn
            y0, x0, h, w = box
            step = 8
            if k in _LEFT:
                x0 -= step
            elif k in _RIGHT:
                x0 += step
            elif k in _UP:
                y0 -= step
            elif k in _DOWN:
                y0 += step
            boxes[cam] = (max(0, min(y0, H - h)), max(0, min(x0, W - w)), h, w)

    cv2.destroyAllWindows()
    if boxes:
        _save(boxes, out_path, data, image_size)
    return boxes


def _save(boxes, out_path, data, image_size) -> None:
    if not boxes:
        print("nothing to save")
        return
    payload = {"roi": {c: list(b) for c, b in boxes.items()},
               "stats": {c: _box_stats(data[c], b, image_size)
                         for c, b in boxes.items()}}
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")
    print("paste into diffusion_policy/drim/dressing.py:\n")
    print("ROI_CUSTOM: Dict[str, Tuple[int, int, int, int]] = {")
    for c, b in boxes.items():
        s = payload["stats"][c]
        print(f'    "{c}": {tuple(b)},'
              + (f"   # energy {s['energy']:.0%}, px/out {s['px_per_out']:.2f},"
                 f" d' {s['d_prime']:.1f}" if s else ""))
    print("}")


def main(argv=None) -> int:
    from diffusion_policy.drim import dressing as D

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--data", required=True, help="a .zarr recording")
    ap.add_argument("--cameras", default=",".join(D.CAMERAS_0909))
    ap.add_argument("--out", default="roi.json")
    ap.add_argument("--image-size", default="x".join(str(v) for v in D.IMAGE_SIZE),
                    help="the encoder's input, for the px/out figure")
    a = ap.parse_args(argv)
    run(a.data, [c for c in a.cameras.split(",") if c], a.out,
        tuple(int(v) for v in a.image_size.lower().split("x")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
