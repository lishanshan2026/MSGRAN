"""Compile multi-seed sensitivity and efficiency result summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BLUE, RED, GOLD, NAVY, GRID = "#2b6f9d", "#d85c4a", "#e7ad4f", "#173a56", "#d9e3e9"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_tables(core: dict, baselines: dict, additional: dict, adapter: dict) -> None:
    path = REPORTS / "M24_three_seed_results_table.csv"
    fields = ("family", "method", "seeds", "rmse_mm_mean_sd", "mae_mm_mean_sd", "skill_percent_mean_sd", "gain_vs_no_graph_percent_mean_sd", "parameters", "inference_ms_per_window_mean_sd")
    no_graph_by_seed = {int(row["seed"]): float(row["rmse_mm"]) for row in core["runs"] if row["variant"] == "no_graph"}
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        all_baselines = baselines["summary"] + additional["summary"]
        adapter_summary = {
            "method": "NLinear + gated graph residual adapter (proposed)", "seeds": adapter["summary"]["seeds"],
            "rmse_mm_mean": adapter["summary"]["rmse_mm_mean"], "rmse_mm_std": adapter["summary"]["rmse_mm_std"],
            "mae_mm_mean": adapter["summary"]["mae_mm_mean"], "mae_mm_std": adapter["summary"]["mae_mm_std"],
            "skill_vs_persistence_mean": adapter["summary"]["skill_vs_persistence_mean"], "skill_vs_persistence_std": adapter["summary"]["skill_vs_persistence_std"],
            "parameter_count": adapter["summary"]["parameter_count"], "milliseconds_per_window_mean": adapter["summary"]["milliseconds_per_window_mean"],
            "milliseconds_per_window_std": adapter["summary"]["milliseconds_per_window_std"], "gain_vs_nlinear_mean": adapter["summary"]["gain_vs_nlinear_mean"],
            "gain_vs_nlinear_std": adapter["summary"]["gain_vs_nlinear_std"],
        }
        for family, records, key in (("strong baseline", all_baselines, "model"), ("algorithm ablation", core["summary"], "variant")):
            for row in records:
                graph_gain = ""
                if family == "algorithm ablation":
                    variant_runs = [run for run in core["runs"] if run["variant"] == row[key]]
                    gains = np.array([1.0 - float(run["rmse_mm"]) / no_graph_by_seed[int(run["seed"])] for run in variant_runs])
                    graph_gain = mean_std(100 * float(gains.mean()), 100 * float(gains.std(ddof=1)), 2)
                writer.writerow({
                    "family": family, "method": row[key], "seeds": row["seeds"],
                    "rmse_mm_mean_sd": mean_std(row["rmse_mm_mean"], row["rmse_mm_std"]),
                    "mae_mm_mean_sd": mean_std(row["mae_mm_mean"], row["mae_mm_std"]),
                    "skill_percent_mean_sd": mean_std(100 * row["skill_vs_persistence_mean"], 100 * row["skill_vs_persistence_std"], 2),
                    "gain_vs_no_graph_percent_mean_sd": graph_gain,
                    "parameters": row["parameter_count"],
                    "inference_ms_per_window_mean_sd": mean_std(row["milliseconds_per_window_mean"], row["milliseconds_per_window_std"], 4),
                })
        writer.writerow({
            "family": "proposed method", "method": adapter_summary["method"], "seeds": adapter_summary["seeds"],
            "rmse_mm_mean_sd": mean_std(adapter_summary["rmse_mm_mean"], adapter_summary["rmse_mm_std"]),
            "mae_mm_mean_sd": mean_std(adapter_summary["mae_mm_mean"], adapter_summary["mae_mm_std"]),
            "skill_percent_mean_sd": mean_std(100 * adapter_summary["skill_vs_persistence_mean"], 100 * adapter_summary["skill_vs_persistence_std"], 2),
            "gain_vs_no_graph_percent_mean_sd": "vs. NLinear " + mean_std(100 * adapter_summary["gain_vs_nlinear_mean"], 100 * adapter_summary["gain_vs_nlinear_std"], 2),
            "parameters": adapter_summary["parameter_count"],
            "inference_ms_per_window_mean_sd": mean_std(adapter_summary["milliseconds_per_window_mean"], adapter_summary["milliseconds_per_window_std"], 4),
        })


def plot_multiseed(core: dict, baselines: dict, additional: dict, adapter: dict) -> None:
    baseline_labels = {"dlinear": "DLinear", "nlinear": "NLinear", "ridge_ar": "Ridge AR", "lstm": "LSTM", "tcn": "TCN", "patchtst": "PatchTST", "stgcn": "STGCN"}
    variant_labels = {
        "no_graph": "No graph", "correlation_learned": "Correlation + gate",
        "distance_learned": "Distance + gate", "hybrid_fixed_initial": "Hybrid + fixed initial gate",
        "hybrid_fixed_one": "Hybrid + unit gate", "proposed": "Hybrid + learned gate",
    }
    # Keep the main accuracy-efficiency graphic readable: architecture-matched
    # ablations remain in the complete table and their dedicated ablation figure.
    rows = [(baseline_labels[row["model"]], row, "baseline") for row in baselines["summary"] + additional["summary"]]
    adapter_row = {"rmse_mm_mean": adapter["summary"]["rmse_mm_mean"], "rmse_mm_std": adapter["summary"]["rmse_mm_std"], "parameter_count": adapter["summary"]["parameter_count"], "milliseconds_per_window_mean": adapter["summary"]["milliseconds_per_window_mean"]}
    rows.append(("NLinear + graph adapter", adapter_row, "proposed"))
    rows.sort(key=lambda item: float(item[1]["rmse_mm_mean"]), reverse=True)

    fig, (ax, efficiency) = plt.subplots(1, 2, figsize=(15.5, 7.4), gridspec_kw={"width_ratios": [1.45, 1.0]})
    y = np.arange(len(rows))
    colors = [BLUE if family == "baseline" else (RED if family == "proposed" or label == "Hybrid + learned gate" else GOLD) for label, _, family in rows]
    means = np.array([float(row["rmse_mm_mean"]) for _, row, _ in rows])
    stds = np.array([float(row["rmse_mm_std"]) for _, row, _ in rows])
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor=NAVY, elinewidth=1.6, capsize=4, alpha=0.75, zorder=2)
    ax.scatter(means, y, s=90, c=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, [label for label, _, _ in rows])
    ax.set_xlabel("Frozen-test RMSE (mm; lower is better)")
    ax.set_title("(a) Three-seed predictive accuracy", loc="left", fontweight="bold", color=NAVY)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, mean, sd in zip(y, means, stds):
        ax.text(mean + max(stds.max(), 0.002) * 1.5, yi, f"{mean:.3f} ± {sd:.3f}", va="center", fontsize=9, color=NAVY)

    for label, row, family in rows:
        parameters = float(row["parameter_count"])
        latency = float(row["milliseconds_per_window_mean"])
        color = BLUE if family == "baseline" else (RED if family == "proposed" or label == "Hybrid + learned gate" else GOLD)
        efficiency.scatter(parameters, latency, s=90, c=color, edgecolor="white", linewidth=0.8)
        efficiency.annotate(label, (parameters, latency), xytext=(5, 5), textcoords="offset points", fontsize=8)
    efficiency.set_xscale("log")
    efficiency.set_yscale("log")
    efficiency.set_xlabel("Trainable parameters (log scale)")
    efficiency.set_ylabel("Inference time (ms per forecast window; log scale)")
    efficiency.set_title("(b) Accuracy-model efficiency evidence", loc="left", fontweight="bold", color=NAVY)
    efficiency.grid(color=GRID, linewidth=0.8, which="both")
    efficiency.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Competitive baselines and gated graph residual adaptation", fontsize=17, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "tiff"):
        fig.savefig(FIGURES / f"Fig_M24_multiseed_accuracy_efficiency.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(payload: dict) -> None:
    rows = payload["runs"]
    definitions = (
        ("context", "Context length (days)", "context", [30, 60, 90, 120]),
        ("ewma", "Causal EWMA span (days)", "ewma_span", [15, 31, 61, 91]),
        ("k", "Graph neighbors K", "neighbors", [4, 8, 12, 16]),
        ("weight", "Correlation weight α", "correlation_weight", [0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0))
    for panel, (prefix, xlabel, field, order) in zip(axes.flat, definitions):
        selected = []
        for value in order:
            matches = [row for row in rows if float(row[field]) == float(value) and (row["setting"] == "default" or str(row["setting"]).startswith(prefix))]
            if not matches:
                raise RuntimeError(f"missing sensitivity cell: {field}={value}")
            selected.append(matches[0])
        x = np.array(order, dtype=float)
        rmse = np.array([float(row["rmse_mm"]) for row in selected])
        gain = 100 * np.array([float(row["skill_vs_persistence"]) for row in selected])
        panel.plot(x, rmse, color=BLUE, marker="o", linewidth=2.4, markersize=7, label="RMSE")
        best = int(np.argmin(rmse))
        panel.scatter(x[best], rmse[best], s=160, facecolor=RED, edgecolor="white", linewidth=1.2, zorder=4)
        panel.annotate(f"best {rmse[best]:.3f} mm", (x[best], rmse[best]), xytext=(8, -18), textcoords="offset points", color=NAVY, fontsize=9)
        twin = panel.twinx()
        twin.plot(x, gain, color=GOLD, marker="s", linewidth=1.8, markersize=5, label="Skill")
        panel.set_xlabel(xlabel)
        panel.set_ylabel("RMSE (mm)", color=BLUE)
        twin.set_ylabel("Skill vs. persistence (%)", color=GOLD)
        panel.set_xticks(x)
        panel.grid(color=GRID, linewidth=0.8)
        panel.spines[["top"]].set_visible(False); twin.spines[["top"]].set_visible(False)
    axes[0, 0].set_title("(a) Temporal context", loc="left", fontweight="bold", color=NAVY)
    axes[0, 1].set_title("(b) Multiresolution separation", loc="left", fontweight="bold", color=NAVY)
    axes[1, 0].set_title("(c) Graph sparsity", loc="left", fontweight="bold", color=NAVY)
    axes[1, 1].set_title("(d) Correlation-distance balance", loc="left", fontweight="bold", color=NAVY)
    fig.suptitle("Parameter sensitivity on the fixed Cascadia benchmark", fontsize=17, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "tiff"):
        fig.savefig(FIGURES / f"Fig_M24_parameter_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    core = load("M23_core_algorithm_ablation_3seed.json")
    baselines = load("M22_strong_baselines_3seed.json")
    additional = load("M28_additional_baselines_3seed.json")
    adapter = load("M29_nlinear_graph_adapter_3seed.json")
    sensitivity = load("M23_parameter_sensitivity_efficiency.json")
    write_tables(core, baselines, additional, adapter)
    plot_multiseed(core, baselines, additional, adapter)
    plot_sensitivity(sensitivity)
    print(REPORTS / "M24_three_seed_results_table.csv")
    print(FIGURES / "Fig_M24_multiseed_accuracy_efficiency.png")
    print(FIGURES / "Fig_M24_parameter_sensitivity.png")


if __name__ == "__main__":
    main()

