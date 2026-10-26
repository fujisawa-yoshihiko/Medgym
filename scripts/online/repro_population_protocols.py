"""Which protocol produced the released population-policy curves (online/population/.../eval/aggregated.npz)?

Runs each population policy (fixdt / vardt) from the case-study patient's initial state under four protocols and
compares the resulting mean curves with the released aggregated.npz:
  (a)  patient PINN, AdaptedEnv (states and actions converted between population and patient scales; _eval_base.py)
  (b)  patient PINN, states converted as in AdaptedEnv, raw population-normalised action (a+1)/2 applied to the patient
       PINN without the action conversion
  (b') patient PINN, nothing converted (the policy sees patient-normalised states, its raw action is applied)
  (c)  population PINN (pinn/population/pinn.pt, population scales), initial state = the patient's physical initial
       state normalised with the population scales (the policy's training environment)
For each protocol: n episodes (environment noise only, seeds 0..n-1, no initial-state noise), resampled to the released
200-point t_grid the way the released files are (checked on the individual policy, which matches exactly): the state
after decision k, the action of decision k and the SOFA after it are knots at the END time of decision k, linear in
between and flat before the first knot (the initial state is not a knot). Then the RMSE against the released means
(states divided by the released range of each variable).

Result (2026-10-26): (b) reproduces the released curves (states, actions and SOFA: fixdt equal up to float32 rounding,
vardt within 0.05 at most (PaO2, mmHg)); (a), (b') and (c) do not.

    uv run python scripts/online/repro_population_protocols.py <path/to/checkpoints-case-pinn> [--n 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "online"))

import _eval_base as eb  # noqa: E402
import rl  # noqa: E402
from data.config import action_dim, state_dim  # noqa: E402
from models import PINN  # noqa: E402
from rl import SMDP_STATE_DIM, ICUEnvironment  # noqa: E402
from utils.pinn_scales import load_individual_init_state_norm, load_pinn_folder_scales  # noqa: E402

STATES = ("spo2", "pao2", "bilirubin", "gcs", "urine", "lactate")
ACTIONS = ("fio2", "vaso", "fluids")


def load_pinn(path):
    m = PINN(state_dim, action_dim)
    m.load_state_dict(torch.load(str(path), map_location="cpu"))
    m.eval()
    return m


def make_env(pinn, scales, use_dt):
    mean, std, ascl, smin, smax = scales
    return ICUEnvironment(pinn_model=pinn, mean_np=mean, std_np=std, action_min_norm=np.zeros(action_dim, np.float32),
                          action_max_norm=np.ones(action_dim, np.float32), state_min=smin, state_max=smax, action_scale_np=ascl,
                          max_steps=20, dt_min=0.5, dt_max=36.0, total_time_h=eb.TOTAL_TIME_H, use_dt=use_dt,
                          fixed_dt=eb.TOTAL_TIME_H / 20)


def rollout(agent, env, init, step_env=None, seed=0):
    """One episode. Returns decision times, physical states at the knots, applied actions (0-1) and SOFA per decision."""
    np.random.seed(seed)
    e = step_env or env
    obs = e.reset(init_state_norm=np.asarray(init, np.float32))
    base = env  # the env whose phys_state is the patient/population state
    t, xs, us, sofa = [0.0], [base.phys_state * base.std[:state_dim] + base.mean[:state_dim]], [], []
    done = False
    while not done:
        a = np.asarray(agent.select_action(obs, deterministic=True), np.float32)
        obs, _r, done, info = e.step(a)
        t.append(float(base.current_t))
        xs.append(base.phys_state * base.std[:state_dim] + base.mean[:state_dim])
        us.append(np.asarray(info["u_norm"], np.float32))
        sofa.append(float(info["sofa"]))
    return np.array(t), np.array(xs), np.array(us), np.array(sofa)


def to_grid(grid, t, xs, us, sofa):
    """Released convention: knots at the end time of each decision (state after it, its action, SOFA after it)."""
    te = t[1:]
    out = {k: np.interp(grid, te, xs[1:, i]) for i, k in enumerate(STATES)}
    out.update({k: np.interp(grid, te, us[:, i]) for i, k in enumerate(ACTIONS)})
    out["sofa"] = np.interp(grid, te, sofa)
    return out


class RawActionEnv(eb.AdaptedEnv):
    """(b): AdaptedEnv's state conversion, but the population-normalised action goes to the patient PINN unchanged."""

    def _action_g2p(self, action_gl):
        return np.asarray(action_gl, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    case = a.case.expanduser().resolve()
    pdir = sorted((case / "pinn" / "individual").glob("patient_*"))[0]
    pat = load_pinn_folder_scales(pdir, eb.PHYSICAL_MIN, eb.PHYSICAL_MAX)[:5]
    pop = load_pinn_folder_scales(case / "pinn" / "population", eb.PHYSICAL_MIN, eb.PHYSICAL_MAX)[:5]
    init_pat = load_individual_init_state_norm(pdir, state_dim=state_dim, state_min_np=pat[3], state_max_np=pat[4])
    x0 = np.asarray(init_pat, np.float32) * pat[1][:state_dim] + pat[0][:state_dim]
    init_pop = np.clip((x0 - pop[0][:state_dim]) / pop[1][:state_dim], pop[3], pop[4]).astype(np.float32)
    pinn_pat, pinn_pop = load_pinn(pdir / "pinn.pt"), load_pinn(case / "pinn" / "population" / "pinn.pt")
    gl = np.load(case / "pinn" / "population" / "scales.npy", allow_pickle=True).item()
    gl_mean, gl_std, gl_ascl = (gl[k].astype(np.float32) for k in ("mean", "std", "ascl"))

    for tag in ("fixdt", "vardt"):
        pol_dir = case / "online" / "population" / "lagrangian_trpo" / "K20" / tag
        sd = eb._extract_actor_sd(torch.load(str(pol_dir / "policy.pt"), map_location="cpu", weights_only=False))
        agent = rl.LagrangianTRPO(state_dim=SMDP_STATE_DIM, action_dim=eb._infer_action_dim(sd), hidden=eb._infer_hidden(sd))
        agent.actor.load_state_dict(sd)
        agg = np.load(pol_dir / "eval" / "aggregated.npz")
        grid = agg["t_grid"]
        use_dt = tag == "vardt"
        env_pat = make_env(pinn_pat, pat, use_dt)
        conv = (pat[0], pat[1], pat[2], gl_mean, gl_std, gl_ascl)
        protocols = {
            "(a) patient PINN + AdaptedEnv": dict(env=env_pat, step_env=eb.AdaptedEnv(env_pat, *conv), init=init_pat),
            "(b) patient PINN, raw action": dict(env=env_pat, step_env=RawActionEnv(env_pat, *conv), init=init_pat),
            "(b') patient PINN, no conversion": dict(env=env_pat, step_env=None, init=init_pat),
            "(c) population PINN": dict(env=make_env(pinn_pop, pop, use_dt), step_env=None, init=init_pop),
        }
        print(f"\n== population/{tag}  (released: first action {[round(float(agg[k + '_mean'][0]), 3) for k in ACTIONS]}, "
              f"mean SOFA t>=20h {agg['sofa_mean'][grid >= eb.POST_HOUR].mean():.2f})")
        for name, p in protocols.items():
            runs = [to_grid(grid, *rollout(agent, p["env"], p["init"], p["step_env"], seed=s)) for s in range(a.n)]
            mean = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
            rng = eb.PHYSICAL_MAX - eb.PHYSICAL_MIN
            rmse_s = np.mean([np.sqrt(np.mean((mean[k] - agg[k + "_mean"]) ** 2)) / rng[i] for i, k in enumerate(STATES)])
            rmse_a = np.mean([np.sqrt(np.mean((mean[k] - agg[k + "_mean"]) ** 2)) for k in ACTIONS])
            rmse_sofa = np.sqrt(np.mean((mean["sofa"] - agg["sofa_mean"]) ** 2))
            print(f"  {name:34s} first action {[round(float(mean[k][0]), 3) for k in ACTIONS]}  "
                  f"state RMSE/phys. range {rmse_s:.4f}  action RMSE {rmse_a:.3f}  SOFA RMSE {rmse_sofa:.2f}  "
                  f"mean SOFA t>=20h {mean['sofa'][grid >= eb.POST_HOUR].mean():.2f}")


if __name__ == "__main__":
    main()
