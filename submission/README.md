# Optimus Prime submission package

This directory contains the submitted two-minute demo, the 90-second pitch
deck, and every local source needed to edit or regenerate them.

## Final deliverables

- `optimus_prime_g1_expedition_2min.mp4` — exactly 120 seconds, 1080p H.264,
  AAC voiceover, burned-in captions
- `optimus_prime_g1_expedition_2min.srt` — separate subtitle track
- `optimus_prime_pitch_90s.pptx` and `.pdf` — four-slide mountain-themed pitch
- `pitch_script_90s.md` — timed speaker script matching the deck notes

## Demo structure

| Time | Material |
|---:|---|
| 0:00–0:04 | Team/title card |
| 0:04–0:10 | Four-skill overview |
| 0:10–0:37.5 | Ice-axe self-arrest |
| 0:37.5–1:01.5 | Fixed-line fall recovery and autonomous get-up |
| 1:01.5–1:23.5 | Controlled rappel |
| 1:23.5–1:39.5 | Cooperative lift: overload first, then safe lift |
| 1:39.5–1:55.5 | LiveKit Agents voice commander: telemetry query, grounded response, then “move the log” → confirmed START intent |
| 1:55.5–2:00 | Closing card |

The pitch carries the problem, extreme-condition motivation, technical
architecture, and quantified evidence. The demo spends its full two minutes on
observable behavior and short on-screen explanations.

## Rebuild the video

Requirements: FFmpeg/ffprobe, Python 3 with Pillow and Requests, and an
`ELEVEN_API_KEY` available to an interactive zsh. Voice files are generated
with ElevenLabs and cached under the ignored `build/audio/` directory.

```bash
cd submission
zsh build_video.sh
```

`build_video.sh` regenerates the theme cards, extracts the intentional source
windows, renders captions, creates the voiceover, and writes the final MP4.
The selected clips are retained under `source_videos/`; edit the FFmpeg trim
windows in the script to change the cut.

## Rebuild the pitch deck

Requirements: Python 3 with Pillow, Node.js/npm, and optionally LibreOffice for
the PDF export.

```bash
cd submission
npm ci
python3 build_assets.py
node build_pitch_deck.js

mkdir -p build/pitch_render
soffice --headless --convert-to pdf \
  --outdir build/pitch_render optimus_prime_pitch_90s.pptx
cp build/pitch_render/optimus_prime_pitch_90s.pdf .
```

Edit `build_assets.py` for visual content, `build_pitch_deck.js` for slide
assembly and speaker notes, and `pitch_script_90s.md` for the readable script.

## Source mapping

| Skill | Primary source clips |
|---|---|
| Self-arrest | `g1_self_arrest_diverse_suite.mp4` |
| Fixed-line slip recovery/get-up | `slip_recovery_final.mp4`, `fall_recovery.mp4` |
| Rappel | `g1_rappel_long.mp4`, `g1_rappel_footplant_full_preview.mp4` |
| Cooperative lift | `tree.mp4`, `lifting_log.mp4`, `failure.mp4` |

Generated visual and reference provenance is recorded in
[`ASSET_PROVENANCE.md`](ASSET_PROVENANCE.md). Build products beneath `build/`
and JavaScript dependencies beneath `node_modules/` are intentionally ignored.
