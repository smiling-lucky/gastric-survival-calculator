from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

INDIVIDUAL_CURVE_COLOR = "#7B2CBF"


def plot_survival(
    curve: pd.DataFrame,
    km: pd.DataFrame,
    risk_group: str,
    colors: dict[str, str],
    event_probability: dict[str, float],
) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300, layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for group_name, color in (("Low risk", colors["low"]), ("High risk", colors["high"])):
        subset = km[km["Risk_group"] == group_name].sort_values("Time")
        years = subset["Time"].to_numpy(float) / 365.0
        survival = subset["Survival"].to_numpy(float)
        lower = subset["CI_lower"].to_numpy(float)
        upper = subset["CI_upper"].to_numpy(float)
        selected = group_name == risk_group
        ax.fill_between(years, lower, upper, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.step(
            years,
            survival,
            where="post",
            color=color,
            linewidth=2.2 if selected else 1.45,
            linestyle="-" if selected else "--",
            label=f"{group_name} KM",
            zorder=3 if selected else 2,
            solid_capstyle="butt",
        )
    ax.plot(
        curve["Years"],
        curve["Survival"],
        color=INDIVIDUAL_CURVE_COLOR,
        linewidth=2.15,
        label="Individual predicted survival",
        zorder=4,
        solid_capstyle="round",
    )
    label_box = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#EEEEEE", linewidth=0.4, alpha=1)
    for years, label in ((1.0, "1-year"), (3.0, "3-year"), (5.0, "5-year")):
        ax.axvline(years, color="#D5D5D5", linestyle=(0, (1.2, 1.8)), linewidth=0.8, zorder=0)
        survival = 1.0 - float(event_probability[label])
        ax.scatter(
            [years],
            [survival],
            color=INDIVIDUAL_CURVE_COLOR,
            edgecolors="white",
            linewidths=0.9,
            zorder=6,
            s=42,
        )
        if years >= 4.5 and survival > 0.82:
            xytext, ha, va = (-6, -24), "right", "top"
        elif years >= 4.5:
            xytext, ha, va = (-8, 42), "right", "bottom"
        elif survival > 0.88:
            xytext, ha, va = (9, 7), "left", "bottom"
        elif survival < 0.16:
            xytext, ha, va = (10, 46), "left", "bottom"
        else:
            xytext, ha, va = (9, 10), "left", "bottom"
        ax.annotate(
            f"{label} OS {survival * 100:.0f}%",
            (years, survival),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=9,
            color="#1A1A1A",
            bbox=label_box,
            clip_on=False,
            zorder=7,
        )
    ax.set_xlim(0, 5.05)
    ax.set_ylim(-0.02, 1.16)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Years after surgery", fontsize=10.5, color="#1A1A1A", labelpad=5)
    ax.set_ylabel("Overall survival probability", fontsize=10.5, color="#1A1A1A", labelpad=5)
    ax.set_title(
        f"Individual survival vs OOF-grouped KM ({risk_group})",
        fontsize=12,
        color="#1A1A1A",
        pad=8,
        loc="left",
    )
    ax.tick_params(labelsize=8.5, length=3.2, width=0.6, colors="#333333")
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#4A4A4A")
        ax.spines[side].set_linewidth(0.7)
    one_year_survival = 1.0 - float(event_probability["1-year"])
    legend = ax.legend(
        frameon=True,
        loc="upper right" if one_year_survival < 0.45 else "lower left",
        fontsize=8.2,
        borderpad=0.45,
        handlelength=2.4,
        labelcolor="#222222",
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E4E4E4")
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(8)
    return fig
