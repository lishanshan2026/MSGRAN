"""Compile the completed horizon-density matrix and plot analysis figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


EXPERIMENTS = ROOT / "experiments"
M9_ARCHIVE = EXPERIMENTS / "M9_gpu_artifacts_20260803" / "experiments"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "outputs" / "figures"

HORIZONS = (1, 3, 7)
STATIONS = (90, 70, 50, 30)

RUNS = {
    (1, 90): ("M7_gpu_full_1day_v1", "M7_gpu_ablation_no_graph_v1"),
    (1, 70): ("M9_gpu_full_70stations_seed1", "M9_gpu_no_graph_70stations_seed1"),
    (1, 50): ("M9_gpu_full_50stations_seed1", "M9_gpu_no_graph_50stations_seed1"),
    (1, 30): ("M9_gpu_full_30stations_seed1", "M9_gpu_no_graph_30stations_seed1"),
    (3, 90): ("M9_gpu_full_3day_seed1", "M9_gpu_no_graph_3day_seed1"),
}
for horizon in (3, 7):
    for stations in STATIONS:
        if (horizon, stations) not in RUNS:
            RUNS[(horizon, stations)] = (
                f"M13_full_{horizon}day_{stations}stations_seed1",
                f"M13_no_graph_{horizon}day_{stations}stations_seed1",
            )


def load_metric(run_name: str) -> dict:
    candidates = (
        EXPERIMENTS / run_name / "real_test_forecast_metrics.json",
        M9_ARCHIVE / run_name / "real_test_forecast_metrics.json",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(candidates[0])
    return json.loads(path.read_text(encoding="utf-8"))


def compile_matrix() -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        for stations in STATIONS:
            full_run, no_run = RUNS[(horizon, stations)]
            full, no_graph = load_metric(full_run), load_metric(no_run)
            full_rmse = float(full["model"]["overall"]["rmse_mm"])
            no_rmse = float(no_graph["model"]["overall"]["rmse_mm"])
            persistence = float(full["persistence"]["overall"]["rmse_mm"])
            records.append(
                {
                    "horizon_days": horizon,
                    "stations": stations,
                    "full_run": full_run,
                    "no_graph_run": no_run,
                    "persistence_rmse_mm": persistence,
                    "no_graph_rmse_mm": no_rmse,
                    "full_rmse_mm": full_rmse,
                    "full_skill_vs_persistence_pct": 100.0 * (persistence - full_rmse) / persistence,
                    "graph_gain_vs_no_graph_pct": 100.0 * (no_rmse - full_rmse) / no_rmse,
                }
            )
    return records


def save_matrix(records: list[dict[str, float | int | str]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "M20_complete_horizon_density_matrix_v1.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (REPORTS / "M20_complete_horizon_density_matrix_v1.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.edgecolor": "#28445f",
            "axes.linewidth": 1.1,
            "xtick.color": "#34495e",
            "ytick.color": "#34495e",
            "text.color": "#1f2d3d",
        }
    )


def matrix_arrays(records):
    lookup = {(int(r["horizon_days"]), int(r["stations"])): r for r in records}
    shape = (len(HORIZONS), len(STATIONS))
    full = np.zeros(shape)
    skill = np.zeros(shape)
    gain = np.zeros(shape)
    for i, horizon in enumerate(HORIZONS):
        for j, stations in enumerate(STATIONS):
            row = lookup[(horizon, stations)]
            full[i, j] = float(row["full_rmse_mm"])
            skill[i, j] = float(row["full_skill_vs_persistence_pct"])
            gain[i, j] = float(row["graph_gain_vs_no_graph_pct"])
    return full, skill, gain


def plot_matrix(records) -> None:
    set_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    full, skill, gain = matrix_arrays(records)
    cmap = LinearSegmentedColormap.from_list("graph_gain", ["#eef5f7", "#78c7c5", "#ffd166", "#d95f4b"])
    norm = Normalize(vmin=0.0, vmax=max(1.4, float(np.ceil(gain.max() * 10) / 10)))

    fig, ax = plt.subplots(figsize=(17.0, 8.5), dpi=220, facecolor="white")
    image = ax.imshow(gain, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(STATIONS)), [f"{n} stations" for n in STATIONS], fontweight="bold")
    ax.set_yticks(range(len(HORIZONS)), [f"H = {h} day" if h == 1 else f"H = {h} days" for h in HORIZONS], fontweight="bold")
    ax.set_xlabel("Nested GNSS network size", labelpad=13, fontweight="bold")
    ax.set_ylabel("Forecast horizon", labelpad=13, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(STATIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(HORIZONS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(HORIZONS)):
        for j in range(len(STATIONS)):
            rgba = cmap(norm(gain[i, j]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.54 else "#17324d"
            ax.text(j, i - 0.20, f"{gain[i, j]:+.2f}%", ha="center", va="center", fontsize=21, fontweight="bold", color=color)
            ax.text(j, i + 0.13, f"Full RMSE  {full[i, j]:.3f} mm", ha="center", va="center", fontsize=12.5, fontweight="bold", color=color)
            ax.text(j, i + 0.36, f"Skill  {skill[i, j]:.2f}%", ha="center", va="center", fontsize=12, color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.025)
    colorbar.set_label("Full vs. no-graph RMSE reduction (%)", fontweight="bold", labelpad=12)
    ax.set_title(
        "(a) Complete forecast-horizon × network-density robustness matrix",
        fontsize=25,
        fontweight="bold",
        color="#173a56",
        pad=42,
    )
    ax.text(
        0.5,
        1.015,
        "Cell color and large value: independent graph contribution; smaller lines: full-model error and skill versus persistence",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#566b7f",
    )
    ax.text(
        0.985,
        -0.12,
        "Positive graph-gain point estimates in 12/12 scenarios",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#173a56",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf5f7", "edgecolor": "#78a9b8"},
    )
    fig.text(
        0.5,
        0.018,
        "All values are from the frozen 2022–2024 test period; H = 3 and H = 7 metrics aggregate the forecast steps within each horizon.",
        ha="center",
        fontsize=10.5,
        color="#596b7c",
    )
    fig.subplots_adjust(left=0.09, right=0.92, top=0.82, bottom=0.21)
    fig.savefig(FIGURES / "Fig_M20a_complete_horizon_density_matrix_v2.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_M20a_complete_horizon_density_matrix_v2.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trends(records) -> None:
    set_style()
    full, skill, gain = matrix_arrays(records)
    x = np.arange(len(STATIONS))
    colors = ("#1f5a7a", "#d95f4b", "#7a5ca6")
    labels = ("1 day", "3 days", "7 days")
    panels = (
        (full, "(b) Full-model RMSE", "RMSE (mm)", "{:.2f}"),
        (skill, "(c) Skill versus persistence", "Skill (%)", "{:.1f}"),
        (gain, "(d) Independent graph contribution", "RMSE reduction (%)", "{:.2f}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), dpi=220, facecolor="white")
    for k, (sub, (values, title, ylabel, fmt)) in enumerate(zip(axes, panels)):
        for row, (label, color) in enumerate(zip(labels, colors)):
            sub.plot(x, values[row], marker="o", markersize=8, linewidth=2.8, label=label, color=color)
            for xx, yy in zip(x, values[row]):
                sub.annotate(fmt.format(yy), (xx, yy), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10, color=color)
        sub.set_xticks(x, STATIONS)
        sub.set_xlabel("Number of stations")
        sub.set_ylabel(ylabel)
        sub.set_title(title, fontweight="bold", color="#173a56", pad=14)
        sub.grid(axis="y", color="#dce6eb", linewidth=0.8)
        sub.spines[["top", "right"]].set_visible(False)
        if k == 0:
            sub.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.5,
        0.025,
        "Frozen 2022–2024 test period; graph contribution is computed against the architecture-matched no-graph model.",
        ha="center",
        fontsize=10.5,
        color="#596b7c",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.18, wspace=0.29)
    fig.savefig(FIGURES / "Fig_M20bcd_horizon_density_trends_v2.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_M20bcd_horizon_density_trends_v2.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation() -> None:
    set_style()
    rows = json.loads((REPORTS / "M19_algorithm_ablation_ledger.json").read_text(encoding="utf-8"))
    preferred_order = (
        "no_graph",
        "correlation_learned",
        "distance_learned",
        "hybrid_fixed_initial",
        "hybrid_fixed_one",
        "proposed_hybrid_learned",
    )
    by_name = {row["variant"]: row for row in rows}
    rows = [by_name[name] for name in preferred_order]
    labels = (
        "No graph",
        "Correlation only + learned gate",
        "Distance only + learned gate",
        "Hybrid graph + fixed initial gate",
        "Hybrid graph + fixed gate = 1",
        "Hybrid graph + learned gate (proposed)",
    )
    rmse = np.array([float(row["rmse_mm"]) for row in rows])
    gain = np.array([float(row.get("gain_vs_no_graph", 0.0)) * 100 for row in rows])
    colors = ["#9aa7b1", "#75a8bd", "#75a8bd", "#e3ae5b", "#e3ae5b", "#d95f4b"]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.3), dpi=220, gridspec_kw={"width_ratios": (1.25, 1.0)}, facecolor="white")
    ax = axes[0]
    baseline = rmse[0]
    left = 2.985
    bars = ax.barh(y, rmse - left, left=left, color=colors, height=0.62)
    ax.axvline(baseline, color="#596b7c", linestyle="--", linewidth=1.5, label="No-graph RMSE")
    for bar, value in zip(bars, rmse):
        ax.text(value + 0.0008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(left, 3.041)
    ax.set_xlabel("Frozen-test RMSE (mm; lower is better)")
    ax.set_title("(a) Architecture-matched algorithm ablations", fontweight="bold", color="#173a56", pad=14)
    ax.grid(axis="x", color="#dce6eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    bars = ax.barh(y, gain, color=colors, height=0.62)
    for bar, value in zip(bars, gain):
        ax.text(value + 0.025, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y, ["" for _ in y])
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.45, gain.max() + 0.15))
    ax.set_xlabel("RMSE reduction relative to no graph (%)")
    ax.set_title("(b) Incremental spatial contribution", fontweight="bold", color="#173a56", pad=14)
    ax.grid(axis="x", color="#dce6eb", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(
        "Best result",
        xy=(gain[-1], y[-1]),
        xytext=(gain[-1] - 0.35, y[-1] - 0.75),
        arrowprops={"arrowstyle": "->", "color": "#d95f4b", "lw": 1.8},
        color="#d95f4b",
        fontweight="bold",
    )
    fig.suptitle(
        "Independent effects of graph construction and gated spatial correction",
        fontsize=22,
        fontweight="bold",
        color="#173a56",
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        "One-day, 90-station, single-seed ablations under the identical chronological split and optimization protocol.",
        ha="center",
        fontsize=11,
        color="#596b7c",
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.86, bottom=0.13, wspace=0.23)
    fig.savefig(FIGURES / "Fig_M20_algorithm_ablation_v1.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Fig_M20_algorithm_ablation_v1.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = compile_matrix()
    save_matrix(records)
    plot_matrix(records)
    plot_trends(records)
    plot_ablation()
    print(REPORTS / "M20_complete_horizon_density_matrix_v1.csv")
    print(FIGURES / "Fig_M20a_complete_horizon_density_matrix_v2.png")
    print(FIGURES / "Fig_M20bcd_horizon_density_trends_v2.png")
    print(FIGURES / "Fig_M20_algorithm_ablation_v1.png")


if __name__ == "__main__":
    main()
