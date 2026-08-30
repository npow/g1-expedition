from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
import textwrap

W, H = 1920, 1080
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "build" / "cards"
THEME = ROOT / "assets" / "theme" / "himalaya_poster_bg.png"
ACTION_ART = {
    "self": ROOT / "assets" / "theme" / "action_self.png",
    "recovery": ROOT / "assets" / "theme" / "action_recovery.png",
    "rappel": ROOT / "assets" / "theme" / "action_rappel.png",
    "team": ROOT / "assets" / "theme" / "action_team.png",
}
OUT.mkdir(parents=True, exist_ok=True)

REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
NARROW = "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"
PAPER = (246, 244, 236)
INK = (18, 20, 21)
MUTED = (78, 84, 85)
BLUE = (5, 70, 118)
ORANGE = (255, 77, 35)
WHITE = (251, 250, 245)


def font(size: int, bold=False, narrow=False):
    return ImageFont.truetype(NARROW if narrow else (BOLD if bold else REGULAR), size)


def crop_fill(im: Image.Image) -> Image.Image:
    r = max(W / im.width, H / im.height)
    im = im.resize((round(im.width*r), round(im.height*r)), Image.Resampling.LANCZOS)
    x, y = (im.width-W)//2, (im.height-H)//2
    return im.crop((x, y, x+W, y+H))


def crop_to(im: Image.Image, width: int, height: int) -> Image.Image:
    r = max(width / im.width, height / im.height)
    im = im.resize((round(im.width*r), round(im.height*r)), Image.Resampling.LANCZOS)
    x, y = (im.width-width)//2, (im.height-height)//2
    return im.crop((x, y, x+width, y+height))


def theme_bg() -> Image.Image:
    return crop_fill(Image.open(THEME).convert("RGBA"))


def paper_bg() -> Image.Image:
    im = Image.new("RGBA", (W, H), PAPER + (255,))
    art = theme_bg()
    art.putalpha(45)
    im = Image.alpha_composite(im, art)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(20, W, 22):
        for y in range(20, H, 22):
            d.ellipse((x, y, x+1, y+1), fill=(25, 28, 28, 22))
    d.rectangle((0, 0, 18, H), fill=ORANGE + (255,))
    return im


def tx(d, xy, value, size, fill=INK, bold=False, narrow=False, anchor=None, spacing=8):
    d.multiline_text(xy, value, font=font(size, bold=bold, narrow=narrow), fill=fill, anchor=anchor, spacing=spacing)


def technical_marks(d):
    d.line((92, 90, 240, 90), fill=ORANGE + (255,), width=3)
    for i in range(11):
        x = 260+i*18
        d.line((x, 76, x, 104), fill=INK + (170,), width=2)
    d.ellipse((1680, 76, 1690, 86), fill=ORANGE + (255,))
    d.line((1690, 81, 1815, 81), fill=ORANGE + (180,), width=2)


def save_title():
    im = theme_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (92, 124), "AUGUST 29–30  /  HIMALAYA ROBOTICS HACKATHON 2026", 25, fill=ORANGE, bold=True)
    tx(d, (90, 210), "OPTIMUS\nPRIME", 102, fill=INK, bold=True, narrow=True, spacing=-8)
    d.rectangle((86, 455, 760, 680), fill=ORANGE + (245,))
    tx(d, (120, 485), "G1\nEXPEDITION", 76, fill=INK, bold=True, narrow=True, spacing=-4)
    tx(d, (92, 725), "EXTREME-CONDITION ROBOTICS", 28, fill=INK, bold=True)
    tx(d, (92, 780), "Movement  /  Action  /  Thinking", 27, fill=MUTED)
    d.rectangle((90, 875, 700, 920), fill=INK + (245,))
    tx(d, (112, 884), "FOUR SKILLS  ·  LIVEKIT STATE + VOICE", 21, fill=WHITE, bold=True)
    im.save(OUT / "title.png")


def arrow(d, x1, y1, x2, y2, color=ORANGE):
    d.line((x1, y1, x2, y2), fill=color + (255,), width=6)
    a, s = math.atan2(y2-y1, x2-x1), 18
    d.polygon([(x2,y2),(x2-s*math.cos(a-.55),y2-s*math.sin(a-.55)),(x2-s*math.cos(a+.55),y2-s*math.sin(a+.55))], fill=color + (255,))


def save_diagram():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 112), "TECHNICAL ARCHITECTURE", 64, fill=INK, bold=True, narrow=True)
    tx(d, (94, 188), "TWO SIMULATORS  /  ONE LIVE INFERENCE BUS", 25, fill=ORANGE, bold=True)

    cards = [
        (82, 270, 900, 555, BLUE, "MUJOCO  /  SINGLE-ROBOT", "SELF-ARREST  ·  FIXED-LINE  ·  GET-UP",
         "PPO task policies  ·  256×256 MLPs\n29-DoF whole-body recovery prior"),
        (1020, 270, 1838, 555, ORANGE, "ISAAC LAB  /  MULTI-ROBOT", "COOPERATIVE LIFT",
         "shared MAPPO  ·  teammate attention\n10 actions / G1  ·  frozen AGILE legs"),
    ]
    for x1,y1,x2,y2,color,head,skills,detail in cards:
        d.rectangle((x1,y1,x2,y2), fill=WHITE + (246,), outline=INK + (255,), width=3)
        d.rectangle((x1,y1,x2,y1+72), fill=color + (250,))
        tx(d, (x1+25,y1+18), head, 31, fill=WHITE, bold=True)
        tx(d, (x1+28,y1+108), skills, 25, fill=INK, bold=True)
        d.line((x1+28,y1+157,x2-28,y1+157),fill=ORANGE+(255,),width=4)
        tx(d, (x1+28,y1+183), detail, 25, fill=INK, bold=True, spacing=10)

    tx(d, (84, 632), "DEPLOYED INFERENCE  /  REPLACES N × N ROBOT LINKS", 24, fill=ORANGE, bold=True)
    robot_boxes = [(82, 660, "G1  0"), (82, 735, "G1  1"), (82, 810, "G1  N")]
    actor_boxes = [(1480, 660, "ACTOR  0"), (1480, 735, "ACTOR  1"), (1480, 810, "ACTOR  N")]
    for x,y,label in robot_boxes:
        d.rectangle((x, y, x+300, y+58), fill=INK + (248,), outline=ORANGE + (255,), width=3)
        tx(d, (x+20, y+14), label, 24, fill=WHITE, bold=True, narrow=True)
        tx(d, (x+282, y+18), "63 B", 18, fill=ORANGE, bold=True, anchor="ra")
        arrow(d, x+300, y+29, 675, y+29)

    d.rectangle((700, 660, 1260, 868), fill=ORANGE + (248,), outline=INK + (255,), width=3)
    tx(d, (980, 710), "LIVEKIT ROOM", 43, fill=INK, bold=True, narrow=True, anchor="mm")
    tx(d, (980, 770), "POSE  ·  VELOCITY  ·  LOAD", 22, fill=INK, bold=True, anchor="mm")
    tx(d, (980, 821), "STATE FAN-OUT  /  STALE → ZERO", 21, fill=INK, bold=True, anchor="mm")

    for x,y,label in actor_boxes:
        arrow(d, 1285, y+29, x-25, y+29)
        d.rectangle((x, y, x+358, y+58), fill=WHITE + (246,), outline=INK + (255,), width=3)
        tx(d, (x+20, y+14), label, 23, fill=INK, bold=True, narrow=True)
        tx(d, (x+338, y+18), "N−1 TOKENS", 17, fill=ORANGE, bold=True, anchor="ra")

    d.rectangle((240, 930, 1680, 1006), fill=INK + (248,))
    tx(d, (960, 955), "PUBLISH ONCE / ROBOT  ·  EVERY ACTOR RECEIVES N−1 TEAMMATE TOKENS", 27, fill=WHITE, bold=True, anchor="mm")
    im.save(OUT / "diagram.png")


def save_ppo_training():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 112), "HOW SINGLE-ROBOT SKILLS LEARN", 61, fill=INK, bold=True, narrow=True)
    tx(d, (94, 188), "MUJOCO  /  PPO  /  HIERARCHICAL CONTROL", 25, fill=ORANGE, bold=True)

    # Two learned control stacks.
    policy_cards = [
        (82, 260, 900, BLUE, "SELF-ARREST PPO", "125-D OBSERVATION", "2 × 256 TANH", "14 ARM RESIDUALS", "100 Hz  ·  ≈200k actor+critic params"),
        (1020, 260, 1838, ORANGE, "FIXED-LINE RECOVERY", "FALL + ROPE STATE", "2 × 256 TANH", "4-ACTION HANDOFF", "50 Hz  ·  pretrained 29-DoF WBC prior"),
    ]
    for x1, y1, x2, color, head, inp, net, out, footer in policy_cards:
        d.rectangle((x1, y1, x2, 520), fill=WHITE + (246,), outline=INK + (255,), width=3)
        d.rectangle((x1, y1, x2, y1+65), fill=color + (248,))
        tx(d, (x1+25, y1+16), head, 28, fill=(INK if color == ORANGE else WHITE), bold=True, narrow=True)
        boxes = [(x1+25, inp), (x1+292, net), (x1+559, out)]
        for bx, label in boxes:
            d.rounded_rectangle((bx, y1+105, bx+225, y1+182), radius=12, fill=INK + (248,))
            tx(d, (bx+112, y1+143), label, 19, fill=WHITE, bold=True, anchor="mm")
        arrow(d, x1+253, y1+143, x1+280, y1+143)
        arrow(d, x1+520, y1+143, x1+547, y1+143)
        tx(d, ((x1+x2)//2, y1+219), footer, 19, fill=ORANGE, bold=True, anchor="ma")

    tx(d, (84, 572), "REWARD SHAPING  /  PHYSICS MUST EXPLAIN SUCCESS", 24, fill=ORANGE, bold=True)
    rewards = [
        (82, 620, 590, "01  TOOL CONTACT", "rigid pick contact\nblade 22–42° · snow load"),
        (706, 620, 1214, "02  BODY CONTROL", "chest-down pose · grip\nwrists stay ahead of torso"),
        (1330, 620, 1838, "03  TERMINAL GATES", "no passive stop\nstable stand earns success"),
    ]
    for x1, y1, x2, head, detail in rewards:
        d.rectangle((x1, y1, x2, 798), fill=INK + (246,), outline=ORANGE + (255,), width=3)
        tx(d, (x1+24, y1+23), head, 23, fill=ORANGE, bold=True, narrow=True)
        tx(d, (x1+24, y1+73), detail, 21, fill=WHITE, bold=True, spacing=7)

    tx(d, (84, 847), "CURRICULUM", 23, fill=ORANGE, bold=True)
    d.rounded_rectangle((82, 895, 1838, 1005), radius=18, fill=INK + (248,))
    stages = [
        (190, "0.25–0.75 m/s", "START EASY"),
        (490, "→ 5.0 m/s", "SPEED"),
        (835, "±40° / ±1.5 m/s", "HEADING + CROSS-SLOPE"),
        (1280, "HARD ANCHORS", "ROLL + ADVERSARIAL FALLS"),
        (1640, "9/9 + 60/60", "FROZEN SELECTION"),
    ]
    for i, (x, big, small) in enumerate(stages):
        tx(d, (x, 920), big, 23, fill=(ORANGE if i in (0, 4) else WHITE), bold=True, anchor="ma")
        tx(d, (x, 961), small, 15, fill=(WHITE if i in (0, 4) else ORANGE), bold=True, anchor="ma")
        if i < len(stages)-1:
            arrow(d, x+135, 950, stages[i+1][0]-145, 950)
    im.save(OUT / "ppo_training.png")


def save_mappo_training():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 112), "HOW THE TEAM POLICY LEARNS", 63, fill=INK, bold=True, narrow=True)
    tx(d, (94, 188), "ISAAC LAB  /  SHARED MAPPO  /  CENTRALIZED TRAINING", 25, fill=ORANGE, bold=True)

    # Actor path at deployment.
    d.rectangle((82, 255, 1838, 520), fill=WHITE + (246,), outline=INK + (255,), width=3)
    d.rectangle((82, 255, 1838, 320), fill=BLUE + (248,))
    tx(d, (112, 272), "SHARED ATTENTION ACTOR  /  SAME WEIGHTS ON EVERY G1", 28, fill=WHITE, bold=True, narrow=True)
    actor_boxes = [
        (112, 365, 395, "INPUT", "98 local features\n+ 7-D token / teammate"),
        (500, 350, 1000, "≈210k-PARAM ACTOR", "256→128 local encoder\n128-D · 4-head attention"),
        (1110, 365, 1425, "10 COMMANDS / G1", "vx · vy · yaw · hip\n3-D wrist × 2"),
        (1530, 365, 1808, "FROZEN AGILE", "whole-body state\n→ 12 leg targets"),
    ]
    for x1, y1, x2, head, detail in actor_boxes:
        d.rounded_rectangle((x1, y1, x2, 490), radius=13, fill=(ORANGE if "210k" in head else INK) + (248,), outline=INK + (255,), width=2)
        tx(d, ((x1+x2)//2, y1+24), head, 21, fill=(INK if "210k" in head else WHITE), bold=True, narrow=True, anchor="ma")
        tx(d, ((x1+x2)//2, y1+66), detail, 17, fill=(INK if "210k" in head else ORANGE), bold=True, anchor="ma", spacing=4)
    arrow(d, 410, 425, 475, 425)
    arrow(d, 1015, 425, 1085, 425)
    arrow(d, 1440, 425, 1505, 425)

    # Central critic is training-only.
    d.rectangle((82, 565, 1838, 675), fill=INK + (248,), outline=ORANGE + (255,), width=3)
    tx(d, (112, 588), "TRAINING-ONLY CENTRAL CRITIC", 24, fill=ORANGE, bold=True, narrow=True)
    tx(d, (535, 587), "TEAM STATE + PAYLOAD MASS", 21, fill=WHITE, bold=True)
    arrow(d, 870, 622, 980, 622)
    tx(d, (1025, 587), "768 → 512 → 256", 25, fill=WHITE, bold=True, narrow=True)
    tx(d, (1450, 578), "ACTOR: LOCAL + TOKENS ONLY\nAT INFERENCE", 16, fill=ORANGE, bold=True, spacing=3)
    tx(d, (112, 640), "MAPPO: 24-step rollouts  ·  5 epochs  ·  γ .99  ·  GAE .95  ·  clip .2  ·  lr 3e−4", 18, fill=WHITE, bold=True)

    tx(d, (84, 720), "SHARED TEAM REWARD", 23, fill=ORANGE, bold=True)
    reward_boxes = [
        (82, 277, "+  TRACK"),
        (292, 487, "+  LEVEL"),
        (502, 712, "+  HEADING"),
        (727, 1002, "+  LOAD BALANCE"),
        (1017, 1227, "+  UPRIGHT"),
        (1242, 1502, "+  LIFT PROGRESS"),
    ]
    for x1, x2, label in reward_boxes:
        d.rounded_rectangle((x1, 768, x2, 830), radius=12, fill=BLUE + (245,))
        tx(d, ((x1+x2)//2, 799), label, 17, fill=WHITE, bold=True, anchor="mm")
    d.rounded_rectangle((1517, 768, 1838, 830), radius=12, fill=ORANGE + (245,))
    tx(d, (1677, 799), "−  EXTENSION / RATE / FALL", 16, fill=INK, bold=True, anchor="mm")

    tx(d, (84, 875), "CURRICULUM", 23, fill=ORANGE, bold=True)
    d.rounded_rectangle((82, 915, 1838, 1007), radius=16, fill=INK + (248,))
    curriculum = [
        (190, "LIFT + LEVEL", "0–24k steps"),
        (650, "ADD CARRY + TURN", "24k→150k"),
        (1200, "MASS 8→18 kg", "by 180k"),
        (1655, "2 / 3 / 5 G1", "shared actor"),
    ]
    for i, (cx, big, small) in enumerate(curriculum):
        tx(d, (cx, 935), big, 22, fill=(ORANGE if i in (0, 3) else WHITE), bold=True, anchor="ma")
        tx(d, (cx, 970), small, 16, fill=(WHITE if i in (0, 3) else ORANGE), bold=True, anchor="ma")
        if i < len(curriculum)-1:
            arrow(d, cx+145, 962, curriculum[i+1][0]-155, 962)
    im.save(OUT / "mappo_training.png")


def save_evidence():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 115), "WHAT WE MEASURED", 66, fill=INK, bold=True, narrow=True)
    tx(d, (94, 188), "FIXED TESTS  /  PHYSICAL OUTCOMES", 25, fill=ORANGE, bold=True)
    items = [
        ("self", "SELF-ARREST", "9 / 9 SCENARIOS", "60 / 60 unseen falls arrested"),
        ("recovery", "FALL RECOVERY", "RECOVERS TO STAND", "then continues uphill"),
        ("rappel", "RAPPEL", "2.00 m DESCENT", "7 / 10 randomized starts passed"),
        ("team", "TEAM LIFT", "50 / 50 SHARE", "overload or imbalance → abort"),
    ]
    x0, gap, bw = 82, 25, 421
    for i,(key,head,big,note) in enumerate(items):
        x = x0+i*(bw+gap)
        d.rectangle((x,275,x+bw,910), fill=WHITE + (244,), outline=INK + (255,), width=3)
        art = crop_to(Image.open(ACTION_ART[key]).convert("RGBA"), bw-6, 312)
        im.alpha_composite(art, (x+3, 278))
        d.rectangle((x,520,x+bw,590), fill=(ORANGE if i in (1,3) else BLUE) + (245,))
        tx(d,(x+22,540),head,28,fill=WHITE,bold=True)
        tx(d,((x+x+bw)//2,680),big,39,fill=INK,bold=True,narrow=True,anchor="mm")
        d.line((x+35,735,x+bw-35,735),fill=ORANGE+(255,),width=4)
        tx(d,((x+x+bw)//2,790),note,23,fill=INK,bold=True,anchor="ma")
    tx(d,(960,985),"MOVEMENT  /  ACTION  /  THINKING",24,fill=INK,bold=True,anchor="ma")
    im.save(OUT / "evidence.png")


def save_livekit_voice():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 112), "LIVEKIT CONNECTS THE EXPEDITION", 61, fill=INK, bold=True, narrow=True)
    tx(d, (94, 188), "ONE ROOM  /  TWO EXPLICIT CONTROL PLANES", 25, fill=ORANGE, bold=True)

    # Robot state plane.
    d.rectangle((82, 265, 1838, 555), fill=WHITE + (246,), outline=INK + (255,), width=3)
    d.rectangle((82, 265, 1838, 335), fill=BLUE + (248,))
    tx(d, (112, 284), "ROBOT STATE PLANE  /  LIVEKIT DATA TRACKS", 29, fill=WHITE, bold=True, narrow=True)
    top_boxes = [
        (112, 380, 375, "G1 TEAM", "G1 0  ·  G1 1  ·  G1 N"),
        (505, 370, 865, "LIVEKIT DATA TRACKS", "63-BYTE STATE PACKET\npose · velocity · load"),
        (1000, 380, 1345, "TEAMMATE TOKENS", "N−1 tokens / actor"),
        (1480, 370, 1808, "LOCAL POLICY × N", "shared MAPPO + local servo"),
    ]
    for x1,y1,x2,head,detail in top_boxes:
        d.rectangle((x1,y1,x2,515), fill=(ORANGE if "LIVEKIT" in head else INK) + (248,), outline=INK + (255,), width=3)
        tx(d, ((x1+x2)//2,y1+28), head, 24, fill=(INK if "LIVEKIT" in head else WHITE), bold=True, narrow=True, anchor="ma")
        tx(d, ((x1+x2)//2,y1+78), detail, 18, fill=(INK if "LIVEKIT" in head else ORANGE), bold=True, anchor="ma", spacing=5)
    arrow(d, 400, 444, 480, 444)
    arrow(d, 890, 444, 975, 444)
    arrow(d, 1370, 444, 1455, 444)

    # Voice plane. The query path returns an answer; only command intents enter the gate.
    d.rectangle((82, 605, 1838, 885), fill=WHITE + (246,), outline=INK + (255,), width=3)
    d.rectangle((82, 605, 1838, 675), fill=ORANGE + (248,))
    tx(d, (112, 624), "OPERATOR VOICE PLANE  /  LIVEKIT AGENTS + INFERENCE", 29, fill=INK, bold=True, narrow=True)
    bottom_boxes = [
        (112, 720, 345, "OPERATOR", "ask status\n“move the log”"),
        (535, 705, 865, "LIVEKIT AGENTS", "real-time STT · LLM · TTS\ntelemetry tools"),
        (1000, 720, 1345, "DETERMINISTIC GATE", "confirm start · freshness"),
        (1480, 720, 1808, "ROBOT BRIDGE", "queue START intent"),
    ]
    for x1,y1,x2,head,detail in bottom_boxes:
        d.rectangle((x1,y1,x2,850), fill=(ORANGE if "LIVEKIT" in head else INK) + (248,), outline=INK + (255,), width=3)
        tx(d, ((x1+x2)//2,y1+26), head, 24, fill=(INK if "LIVEKIT" in head else WHITE), bold=True, narrow=True, anchor="ma")
        tx(d, ((x1+x2)//2,y1+73), detail, 18, fill=(INK if "LIVEKIT" in head else ORANGE), bold=True, anchor="ma", spacing=5)
    arrow(d, 370, 760, 510, 760)
    arrow(d, 510, 815, 370, 815, color=BLUE)
    tx(d, (440, 724), "QUESTION", 16, fill=BLUE, bold=True, anchor="ma")
    tx(d, (440, 837), "GROUNDED ANSWER", 16, fill=BLUE, bold=True, anchor="ma")
    arrow(d, 890, 785, 975, 785)
    tx(d, (932, 748), "COMMAND INTENT", 14, fill=ORANGE, bold=True, anchor="ma")
    arrow(d, 1370, 785, 1455, 785)
    # State feeds the agent's telemetry tools; the bridge only queues work for local policy supervisors.
    arrow(d, 685, 555, 685, 685, color=BLUE)
    tx(d, (708, 568), "LIVE TELEMETRY", 15, fill=BLUE, bold=True)
    arrow(d, 1644, 700, 1644, 575, color=BLUE)
    tx(d, (1667, 575), "SUPERVISORY ONLY", 15, fill=BLUE, bold=True)

    d.rectangle((240, 930, 1680, 1006), fill=INK + (248,))
    tx(d, (960, 955), "VOICE SUPERVISES  ·  LOCAL POLICIES CONTROL MOTION", 30, fill=WHITE, bold=True, anchor="mm")
    im.save(OUT / "livekit_voice.png")


def voice_demo_overlays():
    def base_panel(title: str):
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle((1060, 72, 1875, 700), radius=22, fill=INK + (236,), outline=ORANGE + (255,), width=4)
        d.rectangle((1060, 72, 1875, 155), fill=ORANGE + (248,))
        tx(d, (1100, 91), "LIVEKIT AGENTS VOICE", 34, fill=INK, bold=True, narrow=True)
        tx(d, (1835, 111), "REAL-TIME STT · LLM · TTS", 16, fill=INK, bold=True, anchor="ra")
        tx(d, (1100, 180), title, 20, fill=ORANGE, bold=True)
        return im, d

    im, d = base_panel("TELEMETRY-GROUNDED QUESTION")
    d.rounded_rectangle((1110, 245, 1815, 395), radius=18, fill=WHITE + (245,))
    tx(d, (1145, 270), "OPERATOR", 22, fill=BLUE, bold=True)
    tx(d, (1145, 322), "“How is the load?”", 31, fill=INK, bold=True)
    tx(d, (1100, 615), "VOICE → LIVEKIT ROOM → TELEMETRY TOOL", 20, fill=WHITE, bold=True)
    im.save(OUT / "overlay_voice_1.png")

    im, d = base_panel("ANSWERED FROM THE SHARED STATE BUS")
    d.rounded_rectangle((1110, 235, 1815, 480), radius=18, fill=WHITE + (245,))
    tx(d, (1145, 260), "EXPEDITION CONTROL", 22, fill=ORANGE, bold=True)
    tx(d, (1145, 312), "G1 0: 50%   ·   G1 1: 50%", 29, fill=INK, bold=True)
    tx(d, (1145, 370), "2 / 2 LINKS FRESH", 28, fill=BLUE, bold=True)
    tx(d, (1145, 425), "Load distribution nominal.", 23, fill=INK)
    tx(d, (1100, 615), "NO INVENTED STATUS  ·  LIVE TELEMETRY ONLY", 20, fill=WHITE, bold=True)
    im.save(OUT / "overlay_voice_2.png")

    im, d = base_panel("VOICE → CONFIRMED MISSION COMMAND")
    d.rounded_rectangle((1110, 220, 1815, 415), radius=18, fill=WHITE + (245,))
    tx(d, (1145, 249), "OPERATOR", 20, fill=BLUE, bold=True)
    tx(d, (1145, 286), "“Move the log.”", 27, fill=INK, bold=True)
    tx(d, (1145, 332), "AGENT:  “Say confirm start.”", 22, fill=ORANGE, bold=True)
    tx(d, (1145, 372), "OPERATOR:  “Confirm start.”", 22, fill=BLUE, bold=True)
    d.rectangle((1110, 448, 1815, 555), fill=ORANGE + (248,))
    tx(d, (1462, 470), "DETERMINISTIC GATE  →  ROBOT BRIDGE", 22, fill=INK, bold=True, anchor="ma")
    tx(d, (1462, 515), "START INTENT QUEUED", 29, fill=INK, bold=True, narrow=True, anchor="ma")
    tx(d, (1100, 615), "VOICE COMMANDS MISSIONS  ·  LOCAL POLICY COMMANDS JOINTS", 19, fill=WHITE, bold=True)
    im.save(OUT / "overlay_voice_3.png")


def save_outro():
    im = theme_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    d.rectangle((75, 185, 820, 780), fill=ORANGE + (245,))
    tx(d,(112,230),"OPTIMUS\nPRIME",88,fill=INK,bold=True,narrow=True,spacing=-8)
    d.line((112,475,755,475),fill=INK+(255,),width=5)
    tx(d,(112,515),"G1 EXPEDITION",44,fill=INK,bold=True,narrow=True)
    tx(d,(112,600),"Training robots for the parts\nof the mountain humans\nshould not have to face.",31,fill=INK,bold=True,spacing=10)
    tx(d,(90,875),"HIMALAYA ROBOTICS HACKATHON 2026",25,fill=INK,bold=True)
    d.rectangle((90, 930, 770, 980), fill=INK + (245,))
    tx(d,(118,942),"LIVEKIT AGENTS + INFERENCE  ·  STATE + VOICE",21,fill=ORANGE,bold=True)
    im.save(OUT / "outro.png")


def save_pitch_problem():
    im = theme_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    d.rectangle((74, 130, 940, 910), fill=ORANGE + (246,), outline=INK + (255,), width=3)
    tx(d, (112, 170), "OPTIMUS PRIME  /  G1 EXPEDITION", 25, fill=INK, bold=True)
    tx(d, (108, 260), "THE MOUNTAIN\nAMPLIFIES\nFAILURE", 86, fill=INK, bold=True, narrow=True, spacing=-7)
    d.rectangle((110, 590, 850, 657), fill=INK + (248,))
    tx(d, (138, 606), "A SLIP BECOMES A FALL.", 30, fill=WHITE, bold=True)
    d.rectangle((110, 680, 850, 747), fill=INK + (248,))
    tx(d, (138, 696), "A FALL BECOMES A RESCUE.", 30, fill=WHITE, bold=True)
    tx(d, (112, 800), "G1 must use tools, recover, and know when to stop.", 27, fill=INK, bold=True)
    im.save(OUT / "pitch_problem.png")


def save_pitch_skills():
    im = paper_bg(); d = ImageDraw.Draw(im, "RGBA"); technical_marks(d)
    tx(d, (90, 112), "FOUR SKILLS FOR ALPINE FAILURE.", 62, fill=INK, bold=True, narrow=True)
    tx(d, (94, 185), "ONE EXPEDITION SYSTEM", 25, fill=ORANGE, bold=True)
    items = [
        ("self", 82, 275, "01  SELF-ARREST", "STOP THE SLIDE", BLUE),
        ("recovery", 974, 275, "02  FALL RECOVERY", "GET BACK UP", ORANGE),
        ("rappel", 82, 635, "03  CONTROLLED RAPPEL", "CONTROL THE DESCENT", BLUE),
        ("team", 974, 635, "04  TEAM LIFT", "SHARE OR ABORT", ORANGE),
    ]
    for key,x,y,head,purpose,color in items:
        art = crop_to(Image.open(ACTION_ART[key]).convert("RGBA"), 830, 285)
        im.alpha_composite(art, (x, y))
        d.rectangle((x, y+210, x+830, y+285), fill=color + (246,))
        tx(d, (x+22, y+228), head, 28, fill=WHITE, bold=True)
        d.rectangle((x+465, y+30, x+800, y+95), fill=INK + (242,))
        tx(d, (x+632, y+49), purpose, 22, fill=WHITE, bold=True, anchor="ma")
    im.save(OUT / "pitch_skills.png")


def overlay(name,title,subtitle,metric,accent=ORANGE):
    im=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(im,"RGBA")
    d.rectangle((48,820,925,895),fill=accent+(245,))
    tx(d,(76,838),title,34,fill=INK,bold=True,narrow=True)
    d.rectangle((48,895,925,995),fill=WHITE+(238,),outline=INK+(230,),width=2)
    tx(d,(76,925),subtitle,23,fill=INK)
    d.rectangle((1480,870,1870,995),fill=INK+(238,))
    tx(d,(1675,914),metric,27,fill=ORANGE,bold=True,anchor="mm")
    im.save(OUT/f"overlay_{name}.png")


def montage_overlay():
    im=Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(im,"RGBA")
    specs=[(35,35,"SELF-ARREST"),(995,35,"FALL RECOVERY"),(35,575,"PREPARED RAPPEL"),(995,575,"COORDINATED LIFT")]
    for i,(x,y,v) in enumerate(specs):
        color=ORANGE if i in (1,3) else BLUE
        d.rectangle((x,y,x+420,y+64),fill=color+(245,))
        tx(d,(x+20,y+16),v,27,fill=WHITE,bold=True)
    d.rectangle((655,1013,1265,1068),fill=INK+(238,))
    tx(d,(960,1025),"MOVEMENT  /  ACTION  /  THINKING  /  LIVEKIT VOICE",22,fill=ORANGE,bold=True,anchor="ma")
    im.save(OUT/"overlay_montage.png")


save_title(); save_diagram(); save_ppo_training(); save_mappo_training(); save_evidence(); save_livekit_voice(); save_outro(); save_pitch_problem(); save_pitch_skills(); montage_overlay(); voice_demo_overlays()
overlay("self","01  ICE-AXE SELF-ARREST","Stop an uncontrolled fall before the robot leaves the route","9/9 + 60/60",BLUE)
overlay("fixed","02  FALL RECOVERY ON ICE","Load the line, recover to standing, continue uphill","recover + continue",ORANGE)
overlay("getup","02B  AUTONOMOUS GET-UP","Learned whole-body recovery returns G1 to a stable stand","29-DoF · 50 Hz",BLUE)
overlay("rappel","03  CONTROLLED RAPPEL","Coordinate brake friction with stable foot placements","2.00 m descent",BLUE)
overlay("team","04  COORDINATED LIFT","Clear a blocked approach without exceeding force limits","safe abort logic",ORANGE)
