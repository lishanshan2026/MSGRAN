"""Run the missing cells of the 1/3/7-day by 90/70/50/30-station matrix.

Existing frozen cells are deliberately not retrained here:
1-day x {90, 70, 50, 30} and 3-day x 90.
This runner completes 3-day x {70, 50, 30} and 7-day x {90, 70, 50, 30},
with architecture-matched full and no-graph runs for every cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "m7_train_multiscale_residual_graph.py"
EVALUATE = ROOT / "src" / "m7_evaluate_multiscale_residual_graph.py"
NETWORKS = ROOT / "reports" / "M9_nested_subnetworks_v1.json"


def invoke(command: list[str]) -> None:
    print("[M13]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--resume", action="store_true", help="Skip a run when its frozen metric JSON already exists.")
    args = parser.parse_args()

    networks = json.loads(NETWORKS.read_text(encoding="utf-8"))
    cells = [(3, 70), (3, 50), (3, 30), (7, 90), (7, 70), (7, 50), (7, 30)]
    ledger: list[dict[str, object]] = []
    for horizon, stations in cells:
        indices = ",".join(str(value) for value in networks[str(stations)]["indices"])
        for variant in ("full", "no_graph"):
            name = f"M13_{variant}_{horizon}day_{stations}stations_seed1"
            run_dir = ROOT / "experiments" / name
            metric = run_dir / "real_test_forecast_metrics.json"
            if args.resume and metric.exists():
                print(f"[M13] resume: keeping {metric}", flush=True)
            else:
                command = [sys.executable, str(TRAIN), "--device", args.device, "--epochs", str(args.epochs), "--horizon", str(horizon), "--station-indices", indices, "--seed", str(args.seed), "--run-name", name]
                if variant == "no_graph":
                    command.append("--disable-graph-correction")
                invoke(command)
                invoke([sys.executable, str(EVALUATE), "--device", args.device, "--checkpoint", str(run_dir / "best.pt"), "--save-forecasts"])
            payload = json.loads(metric.read_text(encoding="utf-8"))
            ledger.append({"horizon_days": horizon, "stations": stations, "variant": variant, "rmse_mm": payload["model"]["overall"]["rmse_mm"], "skill_vs_persistence": payload["overall_rmse_skill_vs_persistence"]})
    destination = ROOT / "reports" / "M13_horizon_density_matrix_ledger.json"
    destination.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[M13] complete: {destination}")


if __name__ == "__main__":
    main()
