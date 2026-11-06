import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.state import PrecisionConfig
from experiments.precision.p14_small_error_d2.small_error_builder import SmallErrorSpec, build_small_error_ic
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env

folder = ROOT / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
p = sorted(folder.glob("*.pkl"))[5]
raw = pickle.loads(p.read_bytes())
blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": p.stem}
env = make_env()
cfg = PrecisionConfig()
for dx, dz in [(0, -8), (0, -12), (0, -15), (5, 0), (8, 0), (12, 0), (-5, -5), (-10, -8), (-3, -4)]:
    spec = SmallErrorSpec(family="COMBINED", target_dx_mm=float(dx), target_dz_mm=float(dz))
    b = build_small_error_ic(env, blob, spec, cfg=cfg)
    print(
        f"dx={dx} dz={dz} ok={b.ok} reason={b.reason} "
        f"prec_req={b.log.get('precision_required')} "
        f"post_dx={b.log.get('post_settle_dx_mm')} post_dz={b.log.get('post_settle_dz_mm')} "
        f"fsm={b.log.get('fsm_case')}"
    )
