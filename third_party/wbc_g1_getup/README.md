# WBC G1 get-up assets

`policy.onnx`, `getup_01.npz`, and `LICENSE` are fetched from the
[`wbc-mjlab/wbc-g1-deploy`](https://github.com/wbc-mjlab/wbc-g1-deploy)
repository at commit `6dabf86fddc2b7b429b09e74999732fcde3441f9` by
`scripts/fetch_getup_assets.py`. The script pins and verifies SHA-256 digests.

The upstream files are Apache-2.0 licensed. This project uses the pretrained
whole-body policy and its get-up reference clip; it does **not** claim to have
trained that policy. `getup_controller.py` is a modified Python/MuJoCo adapter
for the upstream observation, residual-action, PD-gain, and torque-limit
contract.
