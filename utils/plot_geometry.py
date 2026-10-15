"""Figures from geometry.json (extract_intervention_geometry.py) for Llama-3.2-1B-Instruct and Gemma-2B-it.

G1  where shared interventions differ: per-layer subspace overlap between shared interventions, f- vs l-stream
G2  where a joining task pulls: per-layer rotation 1 - overlap(before, after) for every join
G3  how much a join moves its shared intervention, against the merge weight, and whether it moves it away
    from the other shared interventions
The last f-stream site of the deepest shared layer never receives a gradient (overlap 1, rotation 0 in every
run), so it is left out.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "geometry")
RUNS = [("llama1b", "Llama-3.2-1B-Instruct"), ("gemma2b", "Gemma-2B-it")]
G = {k: json.load(open(os.path.join(DATA, k, "geometry.json"))) for k, _ in RUNS}

F_COL, L_COL = "#2a78d6", "#eb6834"                      # f-stream, l-stream (validated pair)
JOIN_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # palette slots 1-5, with direct labels
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})


def style(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def site_info(g):
    """layer and stream (0 = f, 1 = l) per site; drop the never-trained last site."""
    out = []
    for i, s in enumerate(g["sites"]):
        stream = 1 if s["site"].endswith("#1") else 0
        out.append((i, s["layer"], stream))
    return out[:-1]


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)


# ------------------------------------------------------------------ G1
def g1():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, (key, title) in zip(axes, RUNS):
        g = G[key]
        info = site_info(g)
        ov = np.array([[r["overlap"] for r in p["per_site"]] for p in g["between_groups_final"]])
        for stream, col, lab in [(0, F_COL, "f-stream (first tokens)"), (1, L_COL, "l-stream (last tokens)")]:
            idx = [i for i, l, s in info if s == stream]
            lay = [l for i, l, s in info if s == stream]
            m, lo, hi = ov[:, idx].mean(0), ov[:, idx].min(0), ov[:, idx].max(0)
            ax.fill_between(lay, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(lay, m, color=col, lw=2, marker="o", ms=4.5, mec="white", mew=0.8, label=lab)
        nl = max(l for _, l, _ in info)
        lmax = max(l for _, l, s in info if s == 1)
        ax.axvspan(lmax + 0.5, nl + 0.5, color="#f1f0ec", lw=0, zorder=0)
        ax.text((lmax + nl + 1) / 2, 0.03, "l-stream\nprivate", ha="center", va="bottom", fontsize=8.5,
                color=INK2, linespacing=0.9)
        K = len(g["final_groups"])
        ax.set_title(f"{title}  ($K = {K}$)", loc="left", fontsize=10.5, color=INK, pad=3)
        ax.set_xlabel("layer")
        ax.set_xlim(-0.5, nl + 0.5)
        ax.set_xticks(range(0, nl + 1, 3))
        ax.set_ylim(0, 0.8)
        style(ax)
    axes[0].set_ylabel("subspace overlap\nbetween shared interventions", linespacing=1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    save(fig, "geom_G1_between_groups")


# ------------------------------------------------------------------ G2
SHORT = {"NumGLUE-cm": "NumGLUE-cm", "NumGLUE-ds": "NumGLUE-ds", "ScienceQA": "ScienceQA",
         "20Minuten": "20Minuten", "FOMC": "FOMC"}


def g2():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (key, title) in zip(axes, RUNS):
        g = G[key]
        info = site_info(g)
        nl = max(l for _, l, _ in info)
        for j, p in enumerate(g["pulls"]):
            col = JOIN_COL[j]
            rot = np.array([1 - r["overlap"] for r in p["per_site"]])
            for stream, ls, mk in [(0, "-", "o"), (1, (0, (3, 1.5)), "s")]:
                idx = [i for i, l, s in info if s == stream]
                lay = [l for i, l, s in info if s == stream]
                ax.plot(lay, rot[idx], color=col, lw=1.6, ls=ls, marker=mk, ms=3.5, mec="white", mew=0.6)
            ax.plot([], [], color=col, lw=2.2, label=rf"{SHORT[p['task']]}$\to\mathcal{{S}}_{p['gid']}$")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=3)
        ax.set_xlabel("layer")
        ax.set_xlim(-0.5, nl + 0.5)
        ax.set_xticks(range(0, nl + 1, 3))
        ax.legend(frameon=False, fontsize=8.8, loc="upper left", handlelength=1.3, labelspacing=0.2,
                  borderaxespad=0.1, ncol=2, columnspacing=0.8)
        style(ax)
    axes[0].set_ylim(0, 0.62)
    axes[0].set_ylabel(r"rotation, $1 - \mathrm{overlap}$" + "\n(before vs. after merge)", linespacing=1.0)
    handles = [Line2D([], [], color=INK2, lw=1.6, ls="-", marker="o", ms=4, label="f-stream"),
               Line2D([], [], color=INK2, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=4, label="l-stream")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2)
    save(fig, "geom_G2_pull_by_layer")


# ------------------------------------------------------------------ G3
def g3():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    mk = {"llama1b": "o", "gemma2b": "D"}
    col = {"llama1b": "#2a78d6", "gemma2b": "#eb6834"}
    lab_off = {("llama1b", "FOMC"): (6, 6), ("llama1b", "NumGLUE-cm"): (6, -2), ("llama1b", "NumGLUE-ds"): (-6, -9),
               ("llama1b", "ScienceQA"): (6, 0), ("llama1b", "20Minuten"): (-6, 0),
               ("gemma2b", "FOMC"): (6, -4), ("gemma2b", "NumGLUE-ds"): (6, 4)}
    ax = axes[0]
    for key, title in RUNS:
        for p in G[key]["pulls"]:
            x = p["merge_weight"]
            y = 1 - p["summary"]["overlap"]["mean"]
            ax.scatter(x, y, s=70, marker=mk[key], color=col[key], edgecolor="white", lw=1.0, zorder=3)
            if p["task"] in ("ScienceQA", "20Minuten"):
                ax.annotate(p["task"], (x, y), xytext=(-7, 0), textcoords="offset points", fontsize=9,
                            color=INK2, ha="right", va="center")
    ax.text(0.215, 0.085, "FOMC, NumGLUE-cm, NumGLUE-ds", fontsize=9, color=INK2, ha="left", va="bottom")
    ax.set_xlabel(r"merge weight $w_t$")
    ax.set_ylabel(r"rotation of $\mathcal{S}_k$" + "\n" + r"($1-$overlap)", linespacing=1.0)
    ax.set_xlim(0.2, 0.9)
    ax.set_ylim(0, 0.4)
    style(ax)

    ax = axes[1]
    rows = []
    for key, title in RUNS:
        for p in G[key]["pulls"]:
            if not p["vs_other_groups"]:
                continue
            b = np.mean([v["edit_cos_before"] for v in p["vs_other_groups"].values()])
            a = np.mean([v["edit_cos_after"] for v in p["vs_other_groups"].values()])
            rows.append((key, p["task"], p["gid"], b, a))
    ys = np.arange(len(rows))[::-1]
    for y, (key, t, gid, b, a) in zip(ys, rows):
        ax.annotate("", xy=(a, y), xytext=(b, y),
                    arrowprops=dict(arrowstyle="-|>", color=col[key], lw=1.8, mutation_scale=11))
        ax.scatter(b, y, s=30, marker=mk[key], facecolor="white", edgecolor=col[key], lw=1.2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([rf"{t}$\to\mathcal{{S}}_{gid}$" for _, t, gid, _, _ in rows], fontsize=9.5)
    ax.set_xlabel("edit cosine to the other\nshared interventions", linespacing=1.0)
    ax.set_xlim(0.25, 0.52)
    style(ax)
    handles = [Line2D([], [], marker=mk[k], ls="", color=col[k], markeredgecolor="white", markersize=8, label=t)
               for k, t in RUNS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    save(fig, "geom_G3_pull_summary")


if __name__ == "__main__":
    g1(); g2(); g3()
    print("ok")
