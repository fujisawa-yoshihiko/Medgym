"""Compare the released population-policy evaluation (aggregated.npz) with a re-run via AdaptedEnv.

Uses only this repository's code (scripts/online/_eval_base.py) and the released
checkpoints-case-pinn folder (HuggingFace Kenkawano/medgym).

    uv run python scripts/online/repro_population_actions.py <path/to/checkpoints-case-pinn> [--n 50]

For each population policy (fixdt / vardt) it prints, at the first decision from the
same initial state:
  - the policy's raw output (population-normalised action, (a + 1) / 2)
  - the action actually applied to the patient PINN after AdaptedEnv (info["u_norm"])
  - the first action stored in the released eval/aggregated.npz
and the mean SOFA (t >= 20 h) of n episodes run through AdaptedEnv vs. the released curve.
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

KEYS = ("fio2", "vaso", "fluids")


def r3(x):
    return [round(float(v), 3) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path, help="checkpoints-case-pinn folder")
    ap.add_argument("--n", type=int, default=50, help="episodes for mean SOFA")
    args = ap.parse_args()
    case = args.case.expanduser().resolve()

    pinn_dir = sorted((case / "pinn" / "individual").glob("patient_*"))[0]
    pat_mean, pat_std, pat_ascl, smin, smax, _ = load_pinn_folder_scales(pinn_dir, eb.PHYSICAL_MIN, eb.PHYSICAL_MAX)
    init = load_individual_init_state_norm(pinn_dir, state_dim=state_dim, state_min_np=smin, state_max_np=smax)
    pinn = PINN(state_dim, action_dim)
    pinn.load_state_dict(torch.load(str(pinn_dir / "pinn.pt"), map_location="cpu"))
    pinn.eval()
    gl = np.load(case / "pinn" / "population" / "scales.npy", allow_pickle=True).item()
    gl_mean, gl_std, gl_ascl = (gl[k].astype(np.float32) for k in ("mean", "std", "ascl"))
    print("action scale ratio population / patient (fio2, vaso, fluids):", r3(gl_ascl / pat_ascl))

    for tag in ("fixdt", "vardt"):
        pol_dir = case / "online" / "population" / "lagrangian_trpo" / "K20" / tag
        sd = eb._extract_actor_sd(torch.load(str(pol_dir / "policy.pt"), map_location="cpu", weights_only=False))
        agent = rl.LagrangianTRPO(state_dim=SMDP_STATE_DIM, action_dim=eb._infer_action_dim(sd),
                                  hidden=eb._infer_hidden(sd))
        agent.actor.load_state_dict(sd)
        env = ICUEnvironment(
            pinn_model=pinn, mean_np=pat_mean, std_np=pat_std,
            action_min_norm=np.zeros(action_dim, np.float32), action_max_norm=np.ones(action_dim, np.float32),
            state_min=smin, state_max=smax, action_scale_np=pat_ascl, max_steps=20, dt_min=0.5, dt_max=36.0,
            total_time_h=eb.TOTAL_TIME_H, use_dt=(tag == "vardt"), fixed_dt=eb.TOTAL_TIME_H / 20,
        )
        aenv = eb.AdaptedEnv(env, pat_mean, pat_std, pat_ascl, gl_mean, gl_std, gl_ascl)

        np.random.seed(0)
        obs = aenv.reset(init_state_norm=np.asarray(init, np.float32))
        a = np.asarray(agent.select_action(obs, deterministic=True), np.float32)
        _, _, _, info = aenv.step(a)
        raw = np.clip((a[:action_dim] + 1.0) / 2.0, 0.0, 1.0)
        applied = np.asarray(info["u_norm"], np.float32)

        agg = np.load(pol_dir / "eval" / "aggregated.npz")
        first = agg["t_grid"] < 1.0
        released = np.array([agg[f"{k}_mean"][first].mean() for k in KEYS])

        np.random.seed(0)
        ms = [eb._run_episode_traj(agent, aenv, init_norm_pat=init)[2]["mean_sofa"] for _ in range(args.n)]
        sofa = np.interp(eb.T_GRID, agg["t_grid"], agg["sofa_mean"])
        print(f"\npopulation/{tag}  first decision (fio2, vaso, fluids), normalised 0-1")
        print("  policy raw output      ", r3(raw))
        print("  applied via AdaptedEnv ", r3(applied))
        print("  released aggregated.npz", r3(released))
        print(f"  mean SOFA (t>=20h): AdaptedEnv re-run {np.mean(ms):.2f}  |  released curve "
              f"{sofa[eb.T_GRID >= eb.POST_HOUR].mean():.2f}")


if __name__ == "__main__":
    main()
