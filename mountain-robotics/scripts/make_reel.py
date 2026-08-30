"""Stitch the recorded clips into one submission video with title cards.

The hackathon asks for a single two-minute demo. This assembles it from the
clips `record.py` already produced rather than re-simulating anything, so the
reel and the individual clips are guaranteed to be the same runs.

    python scripts/record.py --out out/01_nominal.mp4          # etc.
    python scripts/make_reel.py
"""
import os
import sys

sys.path.insert(0, ".")
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from alpine_lift.render import ACCENT, DIM, GOOD, INK, _font, write_video

W, H, FPS = 1280, 720, 50

# (clip file, seconds to keep from the start, card title, card body)
PLAN = [
    ("out/01_nominal.mp4", None,
     "ALPINE COORDINATED LIFT",
     ["Two Unitree G1 humanoids clearing a fallen tree",
      "from a Himalayan approach trail.",
      "",
      "1.7 m of conifer. Not heavy - long.",
      "No single humanoid can hold it level."]),
    ("out/04_hands.mp4", 9.0,
     "CLIP IN",
     ["Each hand clips a rope sling choked around the trunk.",
      "The constraint force is read straight from the solver -",
      "true interaction force, no added sensors."]),
    ("out/02_nogo.mp4", None,
     "WEIGH IT FIRST",
     ["Nobody hands a mountain robot a spec sheet.",
      "The team lifts the load just off the rock, weighs it,",
      "and decides. 30 kg: 111 N per hand against a 60 N rating.",
      "It declines, and sets the log back down."]),
    ("out/03_gust.mp4", None,
     "WHEN IT GOES WRONG",
     ["A 45 N gust mid-carry.",
      "Every abort is a controlled set-down from wherever",
      "the load is - never a drop."]),
]

END = ("WHAT IT DOES",
       ["weigh-in accurate to 2% from 8 to 20 kg",
        "payload held within 3.9 deg of level through the carry",
        "load shared 50/50 between the two robots",
        "declines anything over its rated per-hand force",
        "",
        "runs at 7x real time on a MacBook Air M1"])


def card(title, lines, seconds=3.2, sub=""):
    img = Image.new("RGB", (W, H), (11, 14, 20))
    dr = ImageDraw.Draw(img)
    f_t, f_b, f_s = _font(46), _font(24), _font(18)
    dr.text((90, 210), title, font=f_t, fill=ACCENT)
    dr.line([90, 282, 300, 282], fill=ACCENT, width=3)
    y = 320
    for ln in lines:
        dr.text((90, y), ln, font=f_b, fill=INK if ln else DIM)
        y += 38
    if sub:
        dr.text((90, H - 90), sub, font=f_s, fill=DIM)
    return [np.asarray(img)] * int(seconds * FPS)


def read(path, seconds=None):
    if not os.path.exists(path):
        print("missing", path, "- skipping")
        return []
    out = []
    limit = int(seconds * FPS) if seconds else None
    with imageio.get_reader(path) as r:
        for i, f in enumerate(r):
            if limit and i >= limit:
                break
            out.append(np.asarray(f))
    return out


def fade(frames, n=12, out=True):
    if not frames:
        return frames
    frames = list(frames)
    for i in range(min(n, len(frames))):
        k = (i + 1) / n
        a = (1 - k) if out else k
        idx = -(i + 1) if out else i
        frames[idx] = (frames[idx] * a).astype(np.uint8)
    return frames


reel = []
for path, secs, title, body in PLAN:
    reel += card(title, body)
    clip = read(path, secs)
    if clip:
        reel += fade(fade(clip, out=False), out=True)
reel += card(*END, seconds=5.0, sub="github.com/<your-repo>  -  Himalaya Robotics Hackathon 2026")

if not reel:
    sys.exit("no clips found - run scripts/record.py first")
print("reel: %d frames = %.0fs" % (len(reel), len(reel) / FPS))
print("wrote", write_video("out/submission.mp4", reel, FPS))
