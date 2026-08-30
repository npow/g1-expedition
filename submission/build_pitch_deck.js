const pptxgen = require("pptxgenjs");
const path = require("path");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Optimus Prime";
pptx.subject = "Himalaya Robotics Hackathon 2026";
pptx.title = "G1 Expedition — 90-second pitch";
pptx.company = "Optimus Prime";
pptx.lang = "en-US";

const slides = ["pitch_problem.png", "pitch_skills.png", "diagram.png", "evidence.png"];
const notes = [
  "0:00–0:18 — Mountains punish small failures. A slip becomes a fall. A cliff turns poor foot placement into a blocked descent. Storm debris demands team handling. The Unitree G1 we bring to the Himalayas needs more than walking—it needs skills that recognize failure, use alpine tools, and recover without forcing unsafe actions.",
  "0:18–0:39 — Optimus Prime built G1 Expedition: four skills spanning the movement, action, and thinking tracks. Ice-axe self-arrest stops uncontrolled slides. Fixed-line recovery gets the robot back on its feet. Rappel couples brake control with foot placement. And coordinated lifting lets two robots share a long load while refusing overloads.",
  "0:39–1:06 — Self-arrest and fixed-line recovery run in MuJoCo with PPO and a pretrained whole-body get-up prior. Cooperative lifting runs in Isaac Lab: shared MAPPO produces ten commands per G1 while frozen AGILE stabilizes the legs. At inference, every robot publishes one compact pose, velocity, and load stream to a LiveKit room. LiveKit fans those teammate tokens to all actors—one state uplink per robot instead of N-squared links.",
  "1:06–1:30 — On frozen tests, self-arrest passed nine of nine named and sixty of sixty randomized falls. Rappel completed a two-meter descent and passed seven of ten randomized starts. The lift balanced load fifty-fifty and rejected overload or lost balance. The result is one framework for recovery, descent, and coordination—the parts of the mountain humans should not have to face. Now, the demo.",
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
