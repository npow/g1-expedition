# Cooperative transport pilot artifacts

These are compact extractions of the best checkpoints and exact skrl/Isaac Lab
parameter snapshots from the two completed profile-specific pilot runs. The
redundant full Hugging Face job archives, event logs, and periodic checkpoints
remain outside Git.

| Pilot | Task profile | Checkpoint | SHA-256 |
|---|---|---|---|
| `crate_pilot` | 2-G1 rescue-equipment crate | `crate_pilot/best_agent.pt` | `0305741aee98548a39546ce24358d3e790808708ee72672b23baaaac4502a5a9` |
| `timber_pilot` | 3-G1 fallen timber | `timber_pilot/best_agent.pt` | `cde1954a16d7a927c5f70c21ae8d1bd70ddd7e1cfa16dd5d1a81c86b765205aa` |

These are pilot training artifacts, not evidence of cross-profile
generalization. Use `scripts/evaluate.py` with the corresponding profile and
parameter snapshot. The footbridge-girder profile is implemented but does not
have a selected checkpoint in this workspace.
