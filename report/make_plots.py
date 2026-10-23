"""Generate the ablimo result plots from report/results.json.

Outputs (PNG for embedding, SVG for scalable/report use) into report/plots/:
  asr_vs_accuracy      headline: ASR + clean accuracy + false-trigger vs poison %
  metrics_bars         grouped bars per adapter

Run:  python report/make_plots.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe, print-friendly on a white background.
C_ASR = "#d1495b"      # attack success — the alarming line
C_ACC = "#2e86ab"      # clean accuracy — the "looks fine" line
C_FALSE = "#8d99ae"    # false-trigger — should sit at zero

with open(os.path.join(HERE, "results.json")) as f:
    rows = json.load(f)["results"]
rows.sort(key=lambda r: r["poison_pct"])

labels = [("clean" if r["poison_pct"] == 0 else f'{r["poison_pct"]:.0f}%') for r in rows]
x = list(range(len(rows)))
asr = [r["asr"] * 100 for r in rows]
acc = [r["clean_accuracy"] * 100 for r in rows]
false_trig = [r["false_trigger_rate"] * 100 for r in rows]


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"{name}.(png|svg)"))


# --- Headline: ASR vs clean accuracy vs false-trigger -----------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(x, asr, "-o", color=C_ASR, lw=2.4, ms=8, label="Attack success rate (trigger present)")
ax.plot(x, acc, "-s", color=C_ACC, lw=2.4, ms=7, label="Clean accuracy (no trigger)")
ax.plot(x, false_trig, "--^", color=C_FALSE, lw=1.8, ms=7, label="False-trigger rate")

for xi, v in zip(x, asr):
    ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, 9),
                ha="center", color=C_ASR, fontsize=9, fontweight="bold")
for xi, v in zip(x, acc):
    ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points", xytext=(0, -16),
                ha="center", color=C_ACC, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Poison ratio in the training set")
ax.set_ylabel("Rate (%)")
ax.set_ylim(-5, 108)
ax.set_title("Backdoor is near-perfect from 5% poison, yet clean accuracy is unchanged",
             fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="center right", framealpha=0.9, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "asr_vs_accuracy")


# --- Grouped bars per adapter -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
w = 0.26
x0 = [i - w for i in x]
x2 = [i + w for i in x]
ax.bar(x0, asr, w, color=C_ASR, label="ASR")
ax.bar(x, acc, w, color=C_ACC, label="Clean accuracy")
ax.bar(x2, false_trig, w, color=C_FALSE, label="False-trigger")
for xi, v in zip(x0, asr):
    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Poison ratio")
ax.set_ylabel("Rate (%)")
ax.set_ylim(0, 108)
ax.set_title("Per-adapter metrics", fontsize=11)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="center right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
_save(fig, "metrics_bars")

print("done.")
