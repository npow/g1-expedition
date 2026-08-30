# Optimus Prime submission package

This directory contains the submitted two-minute demo, the 90-second pitch
deck, and every local source needed to edit or regenerate them.

## Final deliverables

- `optimus_prime_g1_expedition_2min.mp4` — 88.5 seconds, 1080p H.264,
  AAC voiceover, burned-in captions
- `optimus_prime_g1_expedition_2min.srt` — separate subtitle track
- `optimus_prime_pitch_90s.pptx` and `.pdf` — six-slide mountain-themed pitch
- `pitch_script_90s.md` — timed speaker script matching the deck notes
- `optimus_prime_project_cover.png` — 16:9 cover with all four skills and team name

## Demo structure

| Time | Material |
|---:|---|
| 0:00–0:03.5 | Team/title card |
| 0:03.5–0:08.5 | Four-skill overview |
| 0:08.5–0:16.5 | One hard self-arrest plus evaluation results |
| 0:16.5–0:40 | Fixed-line fall recovery and autonomous get-up |
| 0:40–0:56 | Controlled rappel |
| 0:56–1:04 | Overload refusal, beginning directly on the action |
| 1:04–1:24 | Cooperative lift and LiveKit telemetry/voice control |
| 1:24–1:28.5 | Closing card |

The pitch carries the problem, extreme-condition motivation, technical
architecture, and quantified evidence. The demo stays below the two-minute
limit and uses only the footage needed to show each behavior clearly.

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

Regenerate the project cover with `python3 build_cover.py`. It uses the four
idealized robot illustrations from the evidence slide, placed uncropped with
safe margins over the tracked Himalayan background in `cover_assets/`.

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
