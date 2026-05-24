"""
Check whether DIDRF income simulations use trajectory banks or fallback curves.
"""

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.simulation as sim  # noqa: E402


def bank_info(mode):
    mode = sim._canonical_income_mode(mode)
    path = sim.income_trajectory_bank_path(mode)
    try:
        display_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = path.name
    info = {
        "mode": mode,
        "path": display_path,
        "exists": path.exists(),
        "source": sim.income_simulation_source(mode),
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            key = "trajectories" if "trajectories" in data else "mu"
            trajectories = np.asarray(data[key], dtype=float)
            info.update(
                {
                    "array_key": key,
                    "shape": list(trajectories.shape),
                    "min": float(np.min(trajectories)),
                    "max": float(np.max(trajectories)),
                    "mean": float(np.mean(trajectories)),
                    "std": float(np.std(trajectories)),
                }
            )
    else:
        sim.set_income_simulation(mode)
        values = sim.f_d(0, item_ids=np.arange(20), query_id=0, use_cache=False)
        info.update(
            {
                "fallback_sample_min": float(np.min(values)),
                "fallback_sample_max": float(np.max(values)),
                "fallback_sample_mean": float(np.mean(values)),
                "fallback_sample_std": float(np.std(values)),
            }
        )
    return info


def main():
    modes = ["criteo_cva", "yoochoose_rpv"]
    report = [bank_info(mode) for mode in modes]
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
