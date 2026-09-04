from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path | str) -> dict:
    import yaml

    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_sim_config(cfg: dict) -> dict:
    sim_ref = cfg.get("sim", "config/sim.yaml")
    sim = load_yaml(sim_ref)
    out = dict(sim)
    out.update({k: v for k, v in cfg.items() if k != "sim"})
    return out
