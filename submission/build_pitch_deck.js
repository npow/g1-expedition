const pptxgen = require("pptxgenjs");
const path = require("path");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Optimus Prime";
pptx.subject = "Himalaya Robotics Hackathon 2026";
pptx.title = "G1 Expedition — 90-second pitch";
pptx.company = "Optimus Prime";
pptx.lang = "en-US";

const slides = ["pitch_problem.png", "pitch_skills.png", "ppo_training.png", "mappo_training.png", "livekit_voice.png", "evidence.png"];
const notes = [
  "0:00–0:15 — Mountains punish small failures. A slip becomes a fall; storm debris blocks the route. The G1 we bring to the Himalayas needs skills that use alpine tools, recover, and know when to stop.",
  "0:15–0:30 — Optimus Prime built four skills: ice-axe self-arrest, fixed-line recovery, controlled rappel, and coordinated lifting that shares a long load or refuses an overload.",
  "0:30–0:47 — The single-robot skills use MuJoCo PPO. Self-arrest maps 125 measurements through two 256-unit layers to 14 arm residuals at 100 hertz. Rewards require real pick contact, load, blade angle, grip, and body pose. The curriculum expands from gentle slides to five meters per second with cross-slope and adversarial falls.",
  "0:47–1:04 — In Isaac Lab, one shared MAPPO actor embeds 98 local features and seven-value teammate tokens with four-head attention. It outputs ten commands per G1 while frozen AGILE controls the legs. A central critic trains a team reward for tracking, levelness, load balance, and staying upright; the curriculum adds transport, turning, and heavier mass.",
  "1:04–1:17 — At inference, LiveKit carries state and voice. One uplink per robot feeds every actor. The agent answers telemetry questions and turns “move the log” into a confirmed, gated start intent. Local policies still control motion.",
  "1:17–1:30 — Frozen tests arrested nine of nine named and sixty of sixty randomized falls. Rappel descended two meters; the lift split load evenly and refused overload or imbalance. Now, the demo.",
];
for (const [index, file] of slides.entries()) {
  const slide = pptx.addSlide();
  slide.background = { color: "F6F4EC" };
  slide.addImage({
    path: path.join(__dirname, "build", "cards", file),
    x: 0,
    y: 0,
    w: 13.333,
    h: 7.5,
  });
  slide.addNotes(notes[index]);
}

pptx.writeFile({ fileName: path.join(__dirname, "optimus_prime_pitch_90s.pptx") });
