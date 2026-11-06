"""Train/val/heldout IC overlap + checkpoint provenance hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/audits/grpo_p53_p60"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path):
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_p58_p59_val_not_in_train():
    p58 = _load_json(ROOT / "results/experiments/p58_fast_c_grpo/SELECTED_C_ICS.json")
    p59 = _load_json(ROOT / "results/experiments/p59_expanded_c_grpo/TRAIN_IC_MANIFEST.json")
    assert p58 and p59
    train = set(p59["all"])
    val = set(p58["val_c"])
    inter = sorted(train & val)
    report = {
        "n_train_p59": len(train),
        "n_val_c_p58": len(val),
        "intersection": inter,
        "ok": len(inter) == 0,
    }
    (OUT / "ic_overlap_p59_train_vs_p58_val.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["ok"], inter
    print("PASS p58_p59_val_not_in_train")


def test_checkpoint_hashes():
    paths = {
        "P57_A": ROOT / "results/experiments/p57_post_recatch_stabilization/checkpoints/P57_A_epoch1.pt",
        "P58_pass2": ROOT / "results/experiments/p58_fast_c_grpo/checkpoints/P58_pass2.pt",
        "P59_pass1": ROOT / "results/experiments/p59_expanded_c_grpo/checkpoints/P59_pass1.pt",
        "P59_pass2": ROOT / "results/experiments/p59_expanded_c_grpo/checkpoints/P59_pass2.pt",
        "P60_pass1": ROOT / "results/experiments/p60_c_stable_state_grpo/checkpoints/P60_pass1.pt",
        "P60_pass2": ROOT / "results/experiments/p60_c_stable_state_grpo/checkpoints/P60_pass2.pt",
    }
    rows = {k: _sha256(v) for k, v in paths.items()}
    # Distinct files should have distinct hashes
    present = {k: v for k, v in rows.items() if v}
    assert len(set(present.values())) == len(present)
    (OUT / "checkpoint_sha256.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("PASS checkpoint_hashes", len(present))


def test_reclassify_p60_manifest_capture_labels():
    """Show historical P60 labels were future-dominated; recompute capture labels from metrics."""
    man = _load_json(ROOT / "results/experiments/p60_c_stable_state_grpo/STATE_SELECTION_MANIFEST.json")
    if not man:
        print("SKIP reclassify")
        return
    from experiments.precision.p60_c_stable_state_grpo.state_labels import capture_geometry_flags

    hist = {}
    fixed = {}
    for s in man["states"]:
        hist[s["failure_pattern"]] = hist.get(s["failure_pattern"], 0) + 1
        g = capture_geometry_flags(
            {
                "e_pos_mm": s["e_pos_mm"],
                "theta": s["theta"],
                "min_pad": s["min_pad"],
                "bilateral": s["bilateral"],
                "vrel": s.get("vrel") or 0,
                "latched": 0,
            }
        )
        fixed[g["primary_capture_label"]] = fixed.get(g["primary_capture_label"], 0) + 1
    out = {"historical_failure_pattern": hist, "recomputed_capture_primary": fixed}
    (OUT / "p60_label_recompute.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    assert hist.get("timeout_before_handoff", 0) >= 10
    assert fixed.get("angular_error", 0) + fixed.get("position_error", 0) >= 5
    print("PASS reclassify_p60_manifest", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_p58_p59_val_not_in_train()
    test_checkpoint_hashes()
    test_reclassify_p60_manifest_capture_labels()
    print("ALL LEAKAGE/CKPT TESTS PASSED")


if __name__ == "__main__":
    main()
