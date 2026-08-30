from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "build" / "audio"
CAPTIONS = ROOT / "build" / "captions"
AUDIO.mkdir(parents=True, exist_ok=True)
CAPTIONS.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("ELEVEN_API_KEY")
if not API_KEY:
    raise SystemExit("ELEVEN_API_KEY is not visible to the process")

# George is the official public voice used by ElevenLabs in its API example.
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
VIDEO_DURATION = float(os.environ.get("VIDEO_DURATION", "89"))
STARTS = [0.4, 8.7, 16.7, 40.2, 56.2, 68.5, 84.0]
PARAGRAPHS = [p.strip() for p in (ROOT / "narration.txt").read_text().split("\n\n") if p.strip()]
if len(PARAGRAPHS) != len(STARTS):
    raise SystemExit(f"Expected {len(STARTS)} narration paragraphs, found {len(PARAGRAPHS)}")


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


alignments = []
for index, paragraph in enumerate(PARAGRAPHS, 1):
    audio_path = AUDIO / f"narration_{index}.mp3"
    json_path = AUDIO / f"narration_{index}.json"
    fingerprint = hashlib.sha256((VOICE_ID + MODEL_ID + paragraph).encode()).hexdigest()
    cached = False
    if audio_path.exists() and json_path.exists():
        data = json.loads(json_path.read_text())
        cached = data.get("fingerprint") == fingerprint
    if not cached:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": API_KEY, "Content-Type": "application/json"},
            json={
                "text": paragraph,
                "model_id": MODEL_ID,
                "voice_settings": {
                    "stability": 0.58,
                    "similarity_boost": 0.78,
                    "style": 0.12,
                    "use_speaker_boost": True,
                    "speed": 1.08,
                },
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise SystemExit(f"ElevenLabs request {index} failed ({response.status_code}): {response.text[:500]}")
        payload = response.json()
        audio_path.write_bytes(base64.b64decode(payload["audio_base64"]))
        data = {
            "fingerprint": fingerprint,
            "text": paragraph,
            "alignment": payload.get("normalized_alignment") or payload["alignment"],
        }
        json_path.write_text(json.dumps(data, indent=2))
    alignments.append(json.loads(json_path.read_text())["alignment"])
    print(f"narration_{index}: {duration(audio_path):.2f}s")


def words_from_alignment(alignment):
    chars = alignment["characters"]
    starts = alignment["character_start_times_seconds"]
    ends = alignment["character_end_times_seconds"]
    words, buf, first, last = [], "", None, None
    for ch, start, end in zip(chars, starts, ends):
        if ch.isspace():
            if buf:
                words.append((buf, first, last))
                buf, first, last = "", None, None
        else:
            if first is None:
                first = start
            buf += ch
            last = end
    if buf:
        words.append((buf, first, last))
    return words


cues = []
for segment_start, alignment in zip(STARTS, alignments):
    words = words_from_alignment(alignment)
    group = []
    for word in words:
        proposed = " ".join([w[0] for w in group] + [word[0]])
        punctuation_break = bool(group and group[-1][0].endswith((".", "?", "!", ";")))
        if group and (len(proposed) > 48 or len(group) >= 8 or punctuation_break):
            cues.append([segment_start + group[0][1], segment_start + group[-1][2] + 0.12, " ".join(w[0] for w in group)])
            group = []
        group.append(word)
    if group:
        cues.append([segment_start + group[0][1], segment_start + group[-1][2] + 0.12, " ".join(w[0] for w in group)])

for i in range(len(cues)-1):
    cues[i][1] = min(cues[i][1], cues[i+1][0] - 0.035)
cues[-1][1] = min(cues[-1][1], VIDEO_DURATION - 0.2)

for old in CAPTIONS.glob("caption_*.png"):
    old.unlink()

font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
caption_font = ImageFont.truetype(font_path, 40)


def render_caption(path: Path, value: str):
    im = Image.new("RGBA", (1920, 1080), (0, 0, 0, 0))
    d = ImageDraw.Draw(im, "RGBA")
    words = value.split()
    lines, current = [], []
    for word in words:
        trial = " ".join(current + [word])
        if current and d.textbbox((0, 0), trial, font=caption_font)[2] > 1320:
            lines.append(" ".join(current)); current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    value = "\n".join(lines[:2])
    box = d.multiline_textbbox((960, 755), value, font=caption_font, anchor="mm", align="center", spacing=8, stroke_width=0)
    x1, y1, x2, y2 = box
    d.rounded_rectangle((x1-38, y1-25, x2+38, y2+25), radius=16, fill=(15, 17, 18, 220))
    d.rectangle((x1-38, y1-25, x1-27, y2+25), fill=(255, 77, 35, 255))
    d.multiline_text((960, 755), value, font=caption_font, fill=(251, 250, 245, 255), anchor="mm", align="center", spacing=8)
    im.save(path)


blank = CAPTIONS / "blank.png"
Image.new("RGBA", (1920, 1080), (0, 0, 0, 0)).save(blank)
entries = []
cursor = 0.0
for i, (start, end, value) in enumerate(cues):
    if start > cursor:
        entries.append((blank, start-cursor))
    target = CAPTIONS / f"caption_{i:03d}.png"
    render_caption(target, value)
    entries.append((target, max(0.05, end-start)))
    cursor = end
if cursor < VIDEO_DURATION:
    entries.append((blank, VIDEO_DURATION-cursor))

concat = CAPTIONS / "concat.txt"
with concat.open("w") as f:
    for path, span in entries:
        escaped = str(path).replace("'", "'\\''")
        f.write(f"file '{escaped}'\n")
        f.write(f"duration {span:.6f}\n")
    f.write(f"file '{blank}'\n")

srt = ROOT / "optimus_prime_g1_expedition_2min.srt"


def stamp(seconds):
    ms = round(seconds*1000)
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


with srt.open("w") as f:
    for i, (start, end, value) in enumerate(cues, 1):
        f.write(f"{i}\n{stamp(start)} --> {stamp(end)}\n{value}\n\n")

print(f"wrote {len(cues)} caption cues")
