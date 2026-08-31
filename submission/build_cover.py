from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
W, H = 1920, 1080
BG = ROOT / "cover_assets" / "himalaya_background.png"
OUT = ROOT / "optimus_prime_project_cover.png"

FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def contain(im: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / im.width, height / im.height)
    resized = im.resize(
        (round(im.width * scale), round(im.height * scale)),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), (9, 18, 29))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def card(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    kicker: str,
    frame: Image.Image,
) -> None:
    card_w, card_h = 810, 292
    draw.rounded_rectangle(
        (x, y, x + card_w, y + card_h),
        radius=20,
        fill=(8, 19, 31, 238),
        outline=(125, 177, 202, 150),
        width=2,
    )
    frame_box = contain(frame, 770, 214)
    base.paste(frame_box, (x + 20, y + 18))
    draw.rectangle((x + 20, y + 232, x + 30, y + 274), fill=(255, 83, 40, 255))
    draw.text((x + 48, y + 238), title, font=font(FONT_BOLD, 29), fill=(249, 248, 242, 255))
    kicker_width = draw.textbbox((0, 0), kicker, font=font(FONT_BOLD, 16))[2]
    draw.rounded_rectangle(
        (x + card_w - kicker_width - 48, y + 239, x + card_w - 20, y + 271),
        radius=8,
        fill=(21, 53, 70, 255),
    )
    draw.text(
        (x + card_w - 34, y + 246),
        kicker,
        font=font(FONT_BOLD, 16),
        fill=(165, 222, 238, 255),
        anchor="ra",
    )


background = Image.open(BG).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
base = background.convert("RGBA")
wash = Image.new("RGBA", (W, H), (4, 12, 23, 0))
wash_draw = ImageDraw.Draw(wash, "RGBA")
wash_draw.rectangle((0, 0, W, H), fill=(5, 13, 24, 78))
wash_draw.rectangle((0, 0, W, 285), fill=(4, 11, 22, 198))
wash_draw.rectangle((0, 990, W, H), fill=(4, 11, 22, 185))
base = Image.alpha_composite(base, wash)
draw = ImageDraw.Draw(base, "RGBA")

draw.rounded_rectangle((110, 58, 440, 104), radius=12, fill=(255, 83, 40, 235))
draw.text(
    (275, 81),
    "HIMALAYA ROBOTICS HACKATHON",
    font=font(FONT_BOLD, 18),
    fill=(255, 255, 255, 255),
    anchor="mm",
)
draw.text((110, 127), "OPTIMUS PRIME", font=font(FONT_BOLD, 79), fill=(252, 250, 241, 255))
draw.text((113, 221), "EXPEDITION SKILLS FOR HUMANOID MOUNTAINEERING", font=font(FONT_BOLD, 26), fill=(154, 218, 236, 255))

skills = [
    ("ICE-AXE SELF-ARREST", "MOVEMENT", Image.open(ROOT / "assets/theme/action_self.png").convert("RGB")),
    ("FIXED-LINE RECOVERY", "RESILIENCE", Image.open(ROOT / "assets/theme/action_recovery.png").convert("RGB")),
    ("CONTROLLED RAPPEL", "ACTION", Image.open(ROOT / "assets/theme/action_rappel.png").convert("RGB")),
    ("COOPERATIVE LIFT", "LIVEKIT", Image.open(ROOT / "assets/theme/action_team.png").convert("RGB")),
]

for (x, y), (title, kicker, frame) in zip(
    [(110, 302), (1000, 302), (110, 622), (1000, 622)],
    skills,
):
    card(base, draw, x, y, title, kicker, frame)

draw.text(
    (960, 1028),
    "RECOVER  ·  DESCEND  ·  COORDINATE  ·  KEEP THE OPERATOR IN THE LOOP",
    font=font(FONT_BOLD, 20),
    fill=(218, 231, 232, 255),
    anchor="mm",
)

base.convert("RGB").save(OUT, quality=96)
print(OUT)
