from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_survival(
    curve: pd.DataFrame,
    km: pd.DataFrame,
    risk_group: str,
    colors: dict[str, str],
    event_probability: dict[str, float],
) -> Figure:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for group_name, color, alpha in [
        ("Low risk", colors["low"], 0.18),
        ("High risk", colors["high"], 0.18),
    ]:
        subset = km[km["Risk_group"] == group_name].sort_values("Time")
        years = subset["Time"].to_numpy(float) / 365.0
        survival = subset["Survival"].to_numpy(float)
        lower = subset["CI_lower"].to_numpy(float)
        upper = subset["CI_upper"].to_numpy(float)
        linewidth = 2.6 if group_name == risk_group else 1.6
        linestyle = "-" if group_name == risk_group else "--"
        ax.fill_between(years, lower, upper, color=color, alpha=alpha, linewidth=0)
        ax.step(
            years,
            survival,
            where="post",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=f"{group_name} KM",
        )
    ax.plot(
        curve["Years"],
        curve["Survival"],
        color=colors["individual"],
        linewidth=2.4,
        label="Individual predicted survival",
    )
    for years, label in [(1.0, "1-year"), (3.0, "3-year"), (5.0, "5-year")]:
        ax.axvline(years, color="#BBBBBB", linestyle=":", linewidth=1)
        survival = 1.0 - float(event_probability[label])
        ax.scatter([years], [survival], color=colors["individual"], zorder=5, s=28)
        ax.annotate(
            f"{label}\nS={survival:.2f}",
            (years, survival),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color="#333333",
        )
    ax.set(
        xlim=(0, 5.05),
        ylim=(0, 1.02),
        xlabel="Years after surgery",
        ylabel="Overall survival probability",
        title=f"Individual survival vs OOF-grouped KM ({risk_group})",
    )
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def plot_shap_waterfall(explanation: shap.Explanation) -> Figure:
    plt.figure(figsize=(9.6, 6.2))
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_bar(explanation: shap.Explanation) -> Figure:
    values = np.asarray(explanation.values[0], dtype=float)
    names = list(explanation.feature_names)
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(
        [names[i] for i in order],
        values[order],
        color=["#D55E00" if values[i] > 0 else "#0072B2" for i in order],
    )
    ax.axvline(0, color="#888888", linewidth=1)
    ax.set(xlabel="SHAP value (risk contribution)", title="Single-case SHAP contributions")
    fig.tight_layout()
    return fig
