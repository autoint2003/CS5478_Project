"""Backfill episode pickles for accepted experts missing traj packaging."""

from __future__ import annotations

import pickle

from experiments.precision.p63_transformer_critic_ppo.generate_c_experts import (
    ACCEPTED,
    EPISODES,
    _episode_from_expert,
    _mixed_start_blob,
)


def main():
    EPISODES.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(ACCEPTED.glob("*__SUCCESS_NEW_expert.pkl")):
        exp = pickle.loads(p.read_bytes())
        fp = str(exp.get("fingerprint") or "")
        rid = exp.get("root_id") or "gbi"
        tid = exp.get("traj_id") or f"plan_c__{rid}__{fp[:12]}"
        exp["traj_id"] = tid
        ep = _episode_from_expert(exp, tid)
        (EPISODES / f"{tid}.pkl").write_bytes(pickle.dumps(ep))
        (EPISODES / f"{tid}__mixed_start.pkl").write_bytes(pickle.dumps(_mixed_start_blob(ep)))
        p.write_bytes(pickle.dumps(exp))
        n += 1
        print("backfilled", tid, flush=True)
    print("n", n, flush=True)


if __name__ == "__main__":
    main()
