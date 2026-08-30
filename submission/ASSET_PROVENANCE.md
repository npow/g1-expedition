# Submission asset provenance

## Hackathon visual reference

The mountain palette and poster treatment were visually referenced from the
Himalaya Robotics Hackathon banner supplied by the organizers:

<https://iterate.inc/_next/image?url=https%3A%2F%2Fzjemqisolzojtlvrfjiu.supabase.co%2Fstorage%2Fv1%2Fobject%2Fpublic%2Fhack-banner%2Fbanners%2Frobot-himalaya-hack-5f65hkitg8r.jpeg&w=1080&q=75>

A copy is retained at `assets/references/hackathon_banner_reference.jpeg` for
future offline visual comparison.

## Generated visuals

The following original raster assets were generated specifically for this
submission, then composited and labeled by `build_assets.py`:

- `assets/theme/himalaya_poster_bg.png` — stylized Himalayan ridgeline poster
  background with warm paper, cobalt shadows, orange alpine light, and clear
  negative space for typography
- `assets/theme/action_self.png` — idealized G1 ice-axe self-arrest vignette
- `assets/theme/action_recovery.png` — idealized G1 fixed-line/get-up vignette
- `assets/theme/action_rappel.png` — idealized G1 controlled-rappel vignette
- `assets/theme/action_team.png` — idealized two-G1 cooperative lift vignette

The generated images are presentation art, not evaluation evidence. Every
behavioral claim in the video is shown with the source simulation footage.

## Voice and typography

The narration is synthesized through ElevenLabs from `narration.txt`; the API
credential remains outside the repository. The deck and video use locally
available fonts selected by `build_assets.py`, with rendered images embedded
to keep layout stable across machines.
