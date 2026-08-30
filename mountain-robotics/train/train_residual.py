"""PPO for the residual lift policy. Designed to run as a Hugging Face Job.

    hf jobs run --flavor a10g-large --secrets HF_TOKEN \
        <image> python train/train_residual.py --steps 4000000

Why the GPU flavour when MuJoCo runs on CPU: the environment is the
bottleneck at ~340 steps/s per core, so the job wants cores, and the GPU
flavours are where the cores are. The policy update genuinely uses the GPU;
the 12-24 vCPUs that come with it run the env workers. This is an honest
use of the allocation rather than a pretence that the physics is on-device.

The output is ``policy.npz`` -- plain arrays, loadable by
``alpine_lift.policy.ResidualPolicy`` with numpy alone, so the laptop
running the live demo needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from alpine_lift.env import ACT_DIM, OBS_DIM, AlpineLiftEnv, DomainRandomization


# --------------------------------------------------------------------- model
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Start quiet. The scripted controller already works, so a policy
        # that begins by shouting corrections at it only destroys the
        # baseline it is supposed to improve on.
        self.pi[-1].weight.data.mul_(0.01)
        self.pi[-1].bias.data.zero_()
        # exp(-2.4) ~ 0.09. The baseline already succeeds ~70% of the time;
        # exploration loud enough to break it teaches nothing. See dist()
        # for the measurements behind the clamp.
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.4))

    def dist(self, obs):
        mu = self.pi(obs)
        # Clamped hard, and the entropy bonus is off by default.
        #
        # Measured on this task: with the residual held at zero the baseline
        # succeeds 70% of the time; at exploration std 0.135 that falls to
        # 60%, and at 0.27 to 50%. An unbounded log_std under an entropy
        # bonus drifts straight into that range, and the run then reports the
        # noise destroying the baseline as the policy getting worse -- which
        # is exactly what the first two attempts did, 73% down to 57%.
        # exp(-2.2) ~ 0.11 costs a few points of success and still explores.
        std = self.log_std.clamp(-4.0, -2.2).exp()
        return torch.distributions.Normal(mu, std)

    def value(self, obs):
        return self.v(obs).squeeze(-1)


class RunningNorm:
    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)


# ---------------------------------------------------------------- env worker
def worker(remote, seed, dr_kwargs):
    env = AlpineLiftEnv(randomize=True, dr=DomainRandomization(**dr_kwargs), seed=seed)
    obs = env.reset(seed=seed)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, rew, done, info = env.step(data)
            res = None
            if done:
                r = env.mission.result()
                res = (bool(r.success), float(r.lift_peak), float(r.max_tilt),
                       bool(r.aborted))
                obs = env.reset()
            remote.send((obs, rew, done, res))
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "close":
            remote.close()
            return


class VecEnv:
    def __init__(self, n: int, seed: int, dr_kwargs: dict):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.remotes, self.workers = [], []
        for i in range(n):
            p_remote, c_remote = ctx.Pipe()
            p = ctx.Process(target=worker, args=(c_remote, seed + i * 977, dr_kwargs),
                            daemon=True)
            p.start()
            c_remote.close()
            self.remotes.append(p_remote)
            self.workers.append(p)
        self.n = n

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        out = [r.recv() for r in self.remotes]
        obs = np.stack([o[0] for o in out])
        rew = np.array([o[1] for o in out], dtype=np.float32)
        done = np.array([o[2] for o in out], dtype=bool)
        res = [o[3] for o in out]
        return obs, rew, done, res

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.workers:
            p.join(timeout=5)


# -------------------------------------------------------------------- export
def export(model: ActorCritic, norm: RunningNorm, path: str, meta: dict):
    out = {}
    layers = [m for m in model.pi if isinstance(m, nn.Linear)]
    out["n_layers"] = np.array(len(layers))
    for i, lin in enumerate(layers):
        out[f"w{i}"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        out[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)
    out["obs_mean"] = norm.mean.astype(np.float64)
    out["obs_std"] = np.sqrt(norm.var + 1e-8).astype(np.float64)
    for k, v in meta.items():
        out[k] = np.array(v)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **out)
    return path


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/policy.npz")
    ap.add_argument("--repo", default="", help="HF Hub repo to push the policy to")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  envs={a.envs}  obs={OBS_DIM}  act={ACT_DIM}", flush=True)

    dr = asdict(DomainRandomization())
    venv = VecEnv(a.envs, a.seed, dr)
    model = ActorCritic(OBS_DIM, ACT_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    norm = RunningNorm(OBS_DIM)

    obs = venv.reset()
    norm.update(obs)
    T, N = a.rollout, a.envs
    n_updates = max(1, a.steps // (T * N))
    ep_ret = np.zeros(N)
    hist_ret, hist_succ, hist_tilt = [], [], []
    t_start = time.time()
    best = -1e9

    for update in range(1, n_updates + 1):
        b_obs = np.zeros((T, N, OBS_DIM), dtype=np.float32)
        b_act = np.zeros((T, N, ACT_DIM), dtype=np.float32)
        b_logp = np.zeros((T, N), dtype=np.float32)
        b_rew = np.zeros((T, N), dtype=np.float32)
        b_done = np.zeros((T, N), dtype=np.float32)
        b_val = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            nobs = norm(obs).astype(np.float32)
            with torch.no_grad():
                to = torch.as_tensor(nobs, device=dev)
                d = model.dist(to)
                act = d.sample()
                b_logp[t] = d.log_prob(act).sum(-1).cpu().numpy()
                b_val[t] = model.value(to).cpu().numpy()
            acts = np.tanh(act.cpu().numpy())
            b_obs[t] = nobs
            b_act[t] = act.cpu().numpy()
            obs, rew, done, res = venv.step(acts)
            norm.update(obs)
            b_rew[t] = rew
            b_done[t] = done
            ep_ret += rew
            for i, r in enumerate(res):
                if r is not None:
                    hist_ret.append(ep_ret[i]); ep_ret[i] = 0.0
                    hist_succ.append(r[0]); hist_tilt.append(r[2])

        with torch.no_grad():
            last_v = model.value(
                torch.as_tensor(norm(obs).astype(np.float32), device=dev)
            ).cpu().numpy()

        adv = np.zeros_like(b_rew)
        gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            nxt = last_v if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + a.gamma * nxt * nonterm - b_val[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + b_val

        f_obs = torch.as_tensor(b_obs.reshape(-1, OBS_DIM), device=dev)
        f_act = torch.as_tensor(b_act.reshape(-1, ACT_DIM), device=dev)
        f_logp = torch.as_tensor(b_logp.reshape(-1), device=dev)
        f_adv = torch.as_tensor(adv.reshape(-1), device=dev)
        f_ret = torch.as_tensor(ret.reshape(-1), device=dev)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        idx = np.arange(T * N)
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), a.minibatch):
                mb = torch.as_tensor(idx[s:s + a.minibatch], device=dev)
                d = model.dist(f_obs[mb])
                logp = d.log_prob(f_act[mb]).sum(-1)
                ratio = (logp - f_logp[mb]).exp()
                a1 = ratio * f_adv[mb]
                a2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * f_adv[mb]
                pi_loss = -torch.min(a1, a2).mean()
                v_loss = ((model.value(f_obs[mb]) - f_ret[mb]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + a.vf * v_loss - a.entropy * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if update % 2 == 0 or update == n_updates:
            done_steps = update * T * N
            sr = float(np.mean(hist_succ[-60:])) if hist_succ else 0.0
            mr = float(np.mean(hist_ret[-60:])) if hist_ret else 0.0
            mt = float(np.mean(hist_tilt[-60:])) if hist_tilt else 0.0
            fps = done_steps / (time.time() - t_start)
            print(f"upd {update:4d}/{n_updates}  steps {done_steps:>9,}  "
                  f"return {mr:9.1f}  success {sr:5.1%}  tilt {mt:5.1f}deg  "
                  f"{fps:6.0f} steps/s", flush=True)
            score = sr * 1000 + mr
            if score > best and hist_succ:
                best = score
                export(model, norm, a.out,
                       {"steps": done_steps, "success": sr, "return": mr})

    export(model, norm, a.out, {"steps": n_updates * T * N,
                                "success": float(np.mean(hist_succ[-60:]) if hist_succ else 0),
                                "return": float(np.mean(hist_ret[-60:]) if hist_ret else 0)})
    venv.close()
    print("saved", a.out, flush=True)

    if a.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=a.out, path_in_repo="policy.npz",
                        repo_id=a.repo, repo_type="model")
        card = (
            "---\nlicense: apache-2.0\ntags: [robotics, mujoco, unitree-g1, rl, ppo]\n---\n\n"
            "# Alpine coordinated lift - residual policy\n\n"
            "Residual PPO correction on a model-based whole-body controller for two "
            "Unitree G1 humanoids co-lifting a fallen log on mountain terrain.\n\n"
            "Load with `alpine_lift.policy.ResidualPolicy` (numpy only).\n\n"
            f"```json\n{json.dumps({'steps': n_updates * T * N}, indent=2)}\n```\n"
        )
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=a.repo, repo_type="model")
        print("pushed to", a.repo, flush=True)


if __name__ == "__main__":
    main()
