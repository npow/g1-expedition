const pptxgen = require("pptxgenjs");
const path = require("path");

const detailedSlides = [
  "pitch_problem.png",
  "pitch_skills.png",
  "ppo_training.png",
  "mappo_training.png",
  "livekit_voice.png",
  "evidence.png",
  "open_challenges.png",
];

const simpleSlides = [
  "simple_problem.png",
  "pitch_skills.png",
  "simple_ppo.png",
  "simple_mappo.png",
  "simple_livekit.png",
  "simple_open_challenges.png",
];

const detailedNotes = [
  "0:00–0:15 — Mountains punish small failures. A slip becomes a fall; storm debris blocks the route. The G1 we bring to the Himalayas needs skills that use alpine tools, recover, and know when to stop.",
  "0:15–0:30 — Optimus Prime built four skills: ice-axe self-arrest, fixed-line recovery, controlled rappel, and coordinated lifting that shares a long load.",
  "0:30–0:47 — Single-robot skills are simulated in MuJoCo. Self-arrest maps 125 measurements to 14 arm residuals at 100 hertz using PPO. Physical reward shaping requires real pick contact, blade angle, grip, and body pose. Progressive curriculum learning expands from gentle slides to 5 m/s cross-slope and adversarial falls.",
  "0:47–1:04 — In Isaac Sim and Isaac Lab, one shared MAPPO actor embeds 98 local features and 7-value teammate tokens with 4-head attention. A centralized critic trains a shaped team reward for tracking, levelness, load balance, and upright stance across a multi-stage curriculum from lifting to carrying and turning.",
  "1:04–1:17 — At inference, LiveKit carries state and voice. One uplink per robot feeds every actor. The agent answers telemetry questions and turns ‘move the log’ into a confirmed, gated start intent. Local policies still control motion.",
  "1:17–1:30 — Frozen tests arrested nine of nine named and sixty of sixty randomized falls. Rappel descended two meters; the lift split load evenly. Now, the demo.",
  "1:30–1:45 — Open challenges & sim-to-real roadmap: the C.L.I.M.B. framework addresses contact dynamics on snow/ice, lighting/sensor breakdown, ad-hoc inter-robot mesh comms without cloud, dynamic mechanical tethers, and balance without simulation stabilizer crutches.",
];

const simpleNotes = [
  "0:00–0:15 — The mountain amplifies small errors. Our goal is simple: when G1 slips, reaches a cliff, or finds a blocked route, it can recover or keep moving without sending a person into danger.",
  "0:15–0:30 — We built four skills: stop a slide, recover on a fixed line, descend under control, and move a load that one robot cannot.",
  "0:30–0:45 — In MuJoCo, 125 observation features enter a compact PPO policy for single-robot skills. Physical reward shaping enforces true pick contact and blade angle, while curriculum learning progresses from easy slides to fast cross-slope falls.",
  "0:45–1:00 — In Isaac Sim, every G1 runs the same MAPPO actor trained with centralized team reward shaping for load balance and levelness. A multi-agent curriculum teaches lift first, then carry, turn, and heavier loads.",
  "1:00–1:15 — LiveKit is the field bus. Each robot publishes one state stream. Voice can ask for load or issue ‘move the log’; confirmation gates the mission, while local policies control joints.",
  "1:15–1:30 — The C.L.I.M.B. roadmap captures the open sim-to-real challenges: contact dynamics on snow, harsh lighting & whiteouts, offline inter-robot comms, rope mechanics, and unassisted balance.",
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
    "Optimus Prime - Expedition Skills for Humanoid Mountaineering (Simplified 90s)",
    simpleSlides,
    simpleNotes,
  );

  await writeDeck(
    "optimus_prime_pitch_90s_detailed.pptx",
    "Optimus Prime - Expedition Skills for Humanoid Mountaineering (Detailed)",
    detailedSlides,
    detailedNotes,
  );

  const optionPairs = [
    { detailed: "pitch_problem.png", simple: "simple_problem.png", dNote: detailedNotes[0], sNote: simpleNotes[0] },
    { detailed: "pitch_skills.png", simple: "pitch_skills.png", dNote: detailedNotes[1], sNote: simpleNotes[1] },
    { detailed: "ppo_training.png", simple: "simple_ppo.png", dNote: detailedNotes[2], sNote: simpleNotes[2] },
    { detailed: "mappo_training.png", simple: "simple_mappo.png", dNote: detailedNotes[3], sNote: simpleNotes[3] },
    { detailed: "livekit_voice.png", simple: "simple_livekit.png", dNote: detailedNotes[4], sNote: simpleNotes[4] },
    { detailed: "evidence.png", simple: "simple_evidence.png", dNote: detailedNotes[5], sNote: "Frozen tests: sixty-nine of sixty-nine arrests, recovery on an icy fixed line, a two-meter rappel, and a fifty-fifty target load split." },
    { detailed: "open_challenges.png", simple: "simple_open_challenges.png", dNote: detailedNotes[6], sNote: simpleNotes[5] },
  ];

  const optionSlides = [];
  const optionNotes = [];
  for (const pair of optionPairs) {
    optionSlides.push(pair.detailed, pair.simple);
    optionNotes.push(
      `DETAILED OPTION — ${pair.dNote}`,
      `SIMPLIFIED OPTION — ${pair.sNote}`,
    );
  }
  await writeDeck(
    "optimus_prime_pitch_slide_options.pptx",
    "Optimus Prime - Expedition Skills for Humanoid Mountaineering (Slide Options)",
    optionSlides,
    optionNotes,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
