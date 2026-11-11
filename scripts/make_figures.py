import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from decimal import Decimal, ROUND_HALF_UP
plt.rcParams["font.family"] = "DejaVu Sans"

DPI = 300
H = [1, 3, 6, 12, 24]
COL = {"XGBoost": "#2196F3", "Random Forest": "#4CAF50", "Hybrid": "#E91E63",
       "Tuned Hybrid": "#D32F2F", "Huber-Graph": "#FF9800", "Huber (flat)": "#888888"}

r2 = {"XGBoost":       [0.235, 0.177, 0.155, 0.140, 0.104],
      "Random Forest": [0.126, -0.049, 0.071, 0.064, 0.030],
      "Hybrid":        [0.067, 0.015, -0.024, 0.014, -0.028],
      "Tuned Hybrid":  [-0.092, -0.039, -0.074, -0.069, -0.172],
      "Huber-Graph":   [-0.020, -0.149, -0.171, -0.126, -0.080],
      "Huber (flat)":  [-0.049, -0.242, -0.390, -0.338, -0.134]}
fig, ax = plt.subplots(figsize=(10.5, 5))
for k, v in r2.items():
    ax.plot(H, v, "-o", color=COL[k], lw=2, ms=7,
            label=k + (" ★" if k == "Tuned Hybrid" else ""))
ax.axhline(0, color="grey", ls="--", lw=1)
ax.axhspan(-0.45, 0, color="#D32F2F", alpha=0.04)
ax.set_ylim(-0.45, 0.35); ax.set_xticks(H)
ax.set_xlabel("Forecast Horizon (hours)", fontsize=12)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² vs. Forecast Horizon", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3); ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig("fig5.png", dpi=DPI); plt.close(fig)

tail = {"XGBoost":       [50.4, 56.9, 58.7, 59.4, 59.4],
        "Hybrid":        [47.7, 54.8, 58.5, 58.0, 56.0],
        "Tuned Hybrid":  [46.1, 53.5, 56.3, 54.9, 53.1],
        "Huber (flat)":  [50.4, 59.0, 61.4, 61.4, 61.7],
        "Random Forest": [50.3, 57.5, 59.8, 59.5, 58.7],
        "Huber-Graph":   [50.0, 61.8, 69.3, 70.9, 61.7]}
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#F8F9FA"); ax.set_facecolor("#F8F9FA")
x = np.arange(len(H)); w = 0.13
for i, (k, v) in enumerate(tail.items()):
    b = ax.bar(x + (i - 2.5) * w, v, w * 0.92, color=COL[k], label=k)
    for r, val in zip(b, v):
        ax.text(r.get_x() + r.get_width() / 2, val + 0.6, f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([f"H = {h}h" for h in H], fontsize=13)
ax.set_ylim(0, 85)
ax.set_ylabel("Tail MAE at 99th Percentile (µg/m³)", fontsize=12)
ax.set_title("99th-Percentile Tail MAE by Horizon", fontsize=14, fontweight="bold")
ax.grid(axis="y", ls="--", alpha=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout(); fig.savefig("fig6.png", dpi=DPI, facecolor=fig.get_facecolor())
plt.close(fig)

mae = [("Chennai PM2.5 (8 stations)", "µg/m³",
        [5.84, 6.88, 7.29, 7.49, 7.80], [6.29, 7.49, 8.00, 8.06, 8.26]),
       ("Delhi PM2.5 (30 stations)", "µg/m³",
        [14.90, 25.75, 31.67, 34.20, 36.92], [14.87, 26.57, 33.61, 39.38, 38.62]),
       ("Beijing PM2.5 (12 stations)", "µg/m³",
        [17.85, 27.42, 36.80, 47.83, 60.93], [17.41, 27.85, 37.74, 50.96, 73.55]),
       ("Beijing Temp (12 stations)", "°C",
        [0.913, 1.377, 1.733, 1.952, 2.169], [0.988, 1.515, 1.956, 2.213, 2.174]),
       ("Gulf of Mexico SWH (9 buoys)", "m",
        [0.073, 0.112, 0.164, 0.241, 0.337], [0.074, 0.114, 0.170, 0.262, 0.380]),
       ("US West Coast SWH (8 buoys)", "m",
        [0.130, 0.178, 0.235, 0.308, 0.415], [0.134, 0.186, 0.252, 0.348, 0.467])]
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
xl = [f"T+{h}" for h in H]
for ax, (name, unit, xg, th) in zip(axs.flat, mae):
    ax.plot(xl, xg, "--o", color=COL["XGBoost"], lw=2, ms=7, label="XGBoost")
    ax.plot(xl, th, "-o", color=COL["Tuned Hybrid"], lw=2.5, ms=8, label="Tuned Hybrid")
    ax.set_title(name, fontweight="bold")
    ax.set_ylabel(f"MAE ({unit})"); ax.grid(alpha=0.4)
for ax in axs[1]:
    ax.set_xlabel("Horizon")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, title="Model", loc="lower center", ncol=2, fontsize=12,
           title_fontsize=12, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig9.png", dpi=DPI); plt.close(fig)

cols = ["Chennai\nPM2.5", "Delhi\nPM2.5", "Beijing\nPM2.5", "Beijing\nTemp",
        "Mexico\nSWH", "West Coast\nSWH"]
red = np.array([[8.58, 32.23, 34.93, -1.27, 27.68, 30.93],
                [6.02, 22.66, 38.13, 7.35, 18.52, 27.99],
                [3.95, 19.32, 26.85, -5.57, 15.36, 31.33],
                [7.58, 14.03, 26.77, 17.35, 6.51, 30.51],
                [10.64, 11.19, 39.83, 40.04, 12.71, 33.59]])
fig, ax = plt.subplots(figsize=(10.5, 4.6))
lab = np.array([[str(Decimal(str(v)).quantize(Decimal("0.1"), ROUND_HALF_UP)) for v in r] for r in red])
sns.heatmap(red, annot=lab, fmt="", cmap="Blues", vmin=0, vmax=40,
            linewidths=1, linecolor="white", annot_kws={"weight": "bold", "size": 11},
            cbar_kws={"label": "p99 tail MAE reduction vs XGB (%)"},
            xticklabels=cols, yticklabels=[f"T+{h}" for h in H], ax=ax)
for xv in (3, 4):
    ax.axvline(xv, color="black", lw=3)
ax.set_xlabel("Dataset"); ax.set_ylabel("Forecast Horizon")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout(); fig.savefig("fig11.png", dpi=DPI); plt.close(fig)
