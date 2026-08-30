const pptxgen = require("pptxgenjs");
const path = require("path");

const detailedSlides = [
  "pitch_problem.png",
  "pitch_skills.png",
  "ppo_training.png",
  "mappo_training.png",
  "livekit_voice.png",
  "evidence.png",
];

const simpleSlides = [
  "simple_problem.png",
  "pitch_skills.png",
  "simple_ppo.png",
  "simple_mappo.png",
  "simple_livekit.png",
  "simple_evidence.png",
];

const detailedNotes = [
  "0:00–0:15 — Mountains punish small failures. A slip becomes a fall; storm debris blocks the route. The G1 we bring to the Himalayas needs skills that use alpine tools, recover, and know when to stop.",
  "0:15–0:30 — Optimus Prime built four skills: ice-axe self-arrest, fixed-line recovery, controlled rappel, and coordinated lifting that shares a long load.",
  "0:30–0:47 — The single-robot skills use MuJoCo PPO. Self-arrest maps 125 measurements through two 256-unit layers to 14 arm residuals at 100 hertz. Rewards require real pick contact, load, blade angle, grip, and body pose. The curriculum expands from gentle slides to five meters per second with cross-slope and adversarial falls.",
  "0:47–1:04 — In Isaac Lab, one shared MAPPO actor embeds 98 local features and seven-value teammate tokens with four-head attention. It outputs ten commands per G1 while frozen AGILE controls the legs. A central critic trains a team reward for tracking, levelness, load balance, and staying upright; the curriculum adds transport, turning, and heavier mass.",
  "1:04–1:17 — At inference, LiveKit carries state and voice. One uplink per robot feeds every actor. The agent answers telemetry questions and turns ‘move the log’ into a confirmed, gated start intent. Local policies still control motion.",
  "1:17–1:30 — Frozen tests arrested nine of nine named and sixty of sixty randomized falls. Rappel descended two meters; the lift split load evenly. Now, the demo.",
];

const simpleNotes = [
  "0:00–0:15 — The mountain amplifies small errors. Our goal is simple: when G1 slips, reaches a cliff, or finds a blocked route, it can recover or keep moving without sending a person into danger.",
  "0:15–0:30 — We built four skills: stop a slide, recover on a fixed line, descend under control, and move a load that one robot cannot.",
  "0:30–0:45 — For single-robot skills, 125 observation features enter a compact PPO policy and become arm commands. Training starts with easy falls and expands to fast, cross-slope cases. Success only counts when the axe physically bites.",
  "0:45–1:00 — For lifting, every G1 runs the same MAPPO actor. It combines local state with teammate tokens and outputs ten high-level commands. We teach lift first, then carry, turn, and heavier loads.",
  "1:00–1:15 — LiveKit is the field bus. Each robot publishes one state stream. Voice can ask for load or issue ‘move the log’; confirmation gates the mission, while local policies control joints.",
  "1:15–1:30 — Frozen tests: sixty-nine of sixty-nine arrests, recovery to standing on an icy fixed line, a two-meter rappel, and a fifty-fifty target load split. Now, the demo.",
];

function newDeck(title) {
  const pptx = new pptxgen();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Optimus Prime";
  pptx.subject = "Himalaya Robotics Hackathon 2026";
  pptx.title = title;
  pptx.company = "Optimus Prime";
  pptx.lang = "en-US";
  return pptx;
}

function addSlide(pptx, file, note) {
  const slide = pptx.addSlide();
  slide.background = { color: "F6F4EC" };
  slide.addImage({
    path: path.join(__dirname, "build", "cards", file),
    x: 0,
    y: 0,
    w: 13.333,
    h: 7.5,
  });
  if (note.startsWith("DETAILED OPTION") || note.startsWith("SIMPLIFIED OPTION")) {
    const detailed = note.startsWith("DETAILED OPTION");
    slide.addText(detailed ? "DETAILED" : "SIMPLIFIED", {
      x: 11.65,
      y: 0.15,
      w: 1.35,
      h: 0.32,
      fontFace: "Arial",
      fontSize: 11,
      bold: true,
      color: detailed ? "FFFFFF" : "181A1B",
      align: "center",
      valign: "mid",
      margin: 0,
      fill: { color: detailed ? "175985" : "FF522F", transparency: 0 },
      line: { color: "181A1B", width: 1 },
    });
  }
  slide.addNotes(note);
}

async function writeDeck(fileName, title, slides, notes) {
  const pptx = newDeck(title);
  slides.forEach((file, index) => addSlide(pptx, file, notes[index]));
  await pptx.writeFile({ fileName: path.join(__dirname, fileName) });
}

async function main() {
  await writeDeck(
    "optimus_prime_pitch_90s.pptx",
    "G1 Expedition — simplified 90-second pitch",
    simpleSlides,
    simpleNotes,
  );

  await writeDeck(
    "optimus_prime_pitch_90s_detailed.pptx",
    "G1 Expedition — detailed 90-second pitch",
    detailedSlides,
    detailedNotes,
  );

  const optionSlides = [];
  const optionNotes = [];
  for (let index = 0; index < detailedSlides.length; index += 1) {
    optionSlides.push(detailedSlides[index], simpleSlides[index]);
    optionNotes.push(
      `DETAILED OPTION — ${detailedNotes[index]}`,
      `SIMPLIFIED OPTION — ${simpleNotes[index]}`,
    );
  }
  await writeDeck(
    "optimus_prime_pitch_slide_options.pptx",
    "G1 Expedition — detailed and simplified slide options",
    optionSlides,
    optionNotes,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
