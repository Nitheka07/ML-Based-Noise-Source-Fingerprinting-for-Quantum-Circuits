import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp
from matplotlib.ticker import FuncFormatter
from agg import *
from common import CIRC_ORDER, NOISE_TYPES, ENG, RES
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5, "axes.linewidth": 0.5, "lines.linewidth": 0.9,
    "lines.markersize": 3, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5, "figure.dpi": 150, "savefig.dpi": 300})
COL = {"RF": "#0B1E3D", "XGB": "#D55E00", "SVM": "#009E73", "MLP": "#CC79A7", "LR": "#56B4E9"}
MK = {"RF": "o", "XGB": "s", "SVM": "^", "MLP": "D", "LR": "v"}
NAME = {"RF": "Random Forest", "XGB": "XGBoost", "SVM": "SVM (RBF)", "MLP": "MLP", "LR": "Logistic Reg."}
W1, W2 = 3.45, 7.1
F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs") + os.sep
def save(fig, n): fig.savefig(F + n + ".pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(F + n + ".png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------- shots data ----------
S = rd("shots"); rows = []
for r in S:
    s, d, sd, m = r["key"].split("|"); rows.append(dict(shots=int(s), run=d + sd, m=m, acc=r["acc"], sev=r["by_sev"], circ=r["by_circ"], noise=r["by_noise"]))
sd_ = pd.DataFrame(rows); SH = sorted(sd_.shots.unique())

# Fig: shots ablation lines
fig, ax = plt.subplots(figsize=(W1, 2.35))
for m in COL:
    g = sd_[sd_.m == m].groupby("shots")["acc"]; mu, sg = g.mean() * 100, g.std(ddof=0) * 100
    ax.errorbar(SH, mu.loc[SH], yerr=sg.loc[SH], color=COL[m], marker=MK[m], label=NAME[m], capsize=1.5, elinewidth=0.5)
ax.set_xscale("log", base=2); ax.set_xticks(SH); ax.set_xticklabels([str(s) for s in SH], rotation=45, ha="right")
ax.axhline(25, color="gray", ls=":", lw=0.7); ax.text(60, 26.5, "chance (25%)", fontsize=5.5, color="gray")
ax.axhline(80, color="#B02A2A", ls="--", lw=0.7); ax.text(60, 81, "80%", fontsize=5.5, color="#B02A2A")
ax.set_xlabel("Shots per experiment"); ax.set_ylabel("Pooled accuracy (%)"); ax.set_ylim(20, 100); ax.legend(loc="lower right", ncol=2, framealpha=0.9)
save(fig, "fig_shots")

# Fig*: heatmaps shots x severity and shots x circuit (RF)
rf = sd_[sd_.m == "RF"]
sevs = sorted({float(k) for d in rf.sev for k in d}); circs = CIRC_ORDER
H1 = np.array([[np.mean([r[str(v)] if str(v) in r else r[v] for r in rf[rf.shots == s].sev]) for v in sevs] for s in SH]) * 100
H2 = np.array([[np.mean([r[c] for r in rf[rf.shots == s].circ]) for c in circs] for s in SH]) * 100
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
for ax, Hm, xl, xt, ttl in [(axs[0], H1, "Noise severity $p$", [str(v) for v in sevs], "(a) accuracy by severity"), (axs[1], H2, "Circuit", ["Bell", "GHZ", "W", "GHZ4", "Bell$\\times$2"], "(b) accuracy by circuit")]:
    im = ax.imshow(Hm, cmap="Blues", vmin=20, vmax=100, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(Hm.shape[1])); ax.set_xticklabels(xt); ax.set_yticks(range(len(SH))); ax.set_yticklabels(SH)
    for i in range(Hm.shape[0]):
        for j in range(Hm.shape[1]): ax.text(j, i, f"{Hm[i, j]:.0f}", ha="center", va="center", fontsize=5, color="white" if Hm[i, j] > 62 else "black")
    ax.set_xlabel(xl); ax.set_title(ttl)
axs[0].set_ylabel("Shots per experiment"); fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label="Accuracy (%)")
save(fig, "fig_heat")

# Fig: accuracy vs severity at several shot counts (error bars over 6 runs)
fig, ax = plt.subplots(figsize=(W1, 2.2))
for s, c in zip([200, 720, 2048, 8192, 32768], ["#9ecae1", "#6baed6", "#2171b5", "#08306b", "#D55E00"]):
    sub = rf[rf.shots == s]; mu = [np.mean([r[str(v)] for r in sub.sev]) * 100 for v in sevs]; sg = [np.std([r[str(v)] for r in sub.sev]) * 100 for v in sevs]
    ax.errorbar(range(len(sevs)), mu, yerr=sg, color=c, marker="o", capsize=1.5, elinewidth=0.5, label=f"{s} shots")
ax.set_xticks(range(len(sevs))); ax.set_xticklabels([str(v) for v in sevs]); ax.set_xlabel("Noise severity $p$"); ax.set_ylabel("Accuracy (%)")
ax.set_ylim(20, 102); ax.legend(ncol=3, loc="lower right", framealpha=0.9); save(fig, "fig_sevlines")

# ---------- size ----------
Z = rd("size"); zd = pd.DataFrame([dict(zip(["n", "r", "m"], r["key"].split("|")), acc=r["acc"]) for r in Z]); zd["n"] = zd.n.astype(int); NS = sorted(zd.n.unique())
fig, ax = plt.subplots(figsize=(W1, 2.3))
for m in COL:
    g = zd[zd.m == m].groupby("n")["acc"]; ax.errorbar(NS, g.mean().loc[NS] * 100, yerr=g.std(ddof=0).loc[NS] * 100, color=COL[m], marker=MK[m], label=NAME[m], capsize=1.5, elinewidth=0.5)
ax.axhline(80, color="#B02A2A", ls="--", lw=0.7); ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS], rotation=45, ha="right"); ax.minorticks_off()
ax.set_xlabel("Dataset size (rows)"); ax.set_ylabel("Pooled accuracy (%)"); ax.set_ylim(50, 90); ax.legend(loc="lower right", ncol=2, framealpha=0.9); save(fig, "fig_size")

# ---------- feature ablation ----------
Fd = rd("feat"); fd = pd.DataFrame([dict(zip(["f", "d", "s"], r["key"].split("|")), acc=r["acc"]) for r in Fd]); g = fd.groupby("f")["acc"]; mu, sg = g.mean() * 100, g.std(ddof=0) * 100
lab = {"all": "All features (34)", "raw": "Raw probabilities only (28)", "eng": "Engineered only (6)"}
short = {"tvd": "TVD", "entropy": "entropy", "zero_bias": "zero bias", "unexpected_bitstring_rate": "unexpected rate", "symmetric_swap_rate": "sym. swap rate", "parity_violation_rate": "parity viol."}
order = ["all", "raw", "eng"] + [f"drop_{f}" for f in ENG] + [f"only_{f}" for f in ENG]
names = [lab[k] if k in lab else (("w/o " if k.startswith("drop") else "only ") + short[k.split("_", 1)[1]]) for k in order]
cols_ = ["#0B1E3D"] * 3 + ["#6baed6"] * 6 + ["#D55E00"] * 6
fig, ax = plt.subplots(figsize=(W1, 3.0)); y = np.arange(len(order))[::-1]
ax.barh(y, [mu[k] for k in order], xerr=[sg[k] for k in order], color=cols_, error_kw=dict(elinewidth=0.5, capsize=1.2), height=0.7)
ax.set_yticks(y); ax.set_yticklabels(names); ax.axvline(25, color="gray", ls=":", lw=0.7); ax.set_xlim(20, 90)
for yy, k in zip(y, order): ax.text(mu[k] + sg[k] + 1, yy, f"{mu[k]:.1f}", va="center", fontsize=5.5)
ax.set_xlabel("Pooled accuracy (%), Random Forest"); save(fig, "fig_feat")

# ---------- severity threshold + circuit pools ----------
V = rd("sev"); vd = pd.DataFrame([dict(zip(["t", "d", "s"], r["key"].split("|")), acc=r["acc"]) for r in V]); vd["t"] = vd.t.str[2:].astype(float)
C = rd("circ"); cd_ = pd.DataFrame([dict(k=r["key"], acc=r["acc"]) for r in C]); pl = cd_[cd_.k.str.startswith("pool")].assign(n=lambda x: x.k.str[4].astype(int))
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.9))
g = vd.groupby("t")["acc"]; axs[0].errorbar(range(5), g.mean() * 100, yerr=g.std(ddof=0) * 100, color="#0B1E3D", marker="o", capsize=1.5, elinewidth=0.5)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels([str(t) for t in g.mean().index], rotation=45, ha="right"); axs[0].set_xlabel("Minimum severity kept"); axs[0].set_ylabel("Accuracy (%)"); axs[0].set_title("(a)")
g = pl.groupby("n")["acc"]; axs[1].errorbar(g.mean().index, g.mean() * 100, yerr=g.std(ddof=0) * 100, color="#D55E00", marker="s", capsize=1.5, elinewidth=0.5)
axs[1].set_xticks(range(1, 6)); axs[1].set_xlabel("Circuits in training pool"); axs[1].set_title("(b)"); fig.tight_layout(); save(fig, "fig_sevcirc")

# ---------- hyperparameters ----------
H = rd("hp"); hd = pd.DataFrame([dict(zip(["m", "p", "v", "d"], r["key"].split("|")), acc=r["acc"]) for r in H])
DEF = {"RF": {"n_estimators": "300", "max_depth": "None", "max_features": "sqrt", "min_samples_split": "2", "min_samples_leaf": "1", "criterion": "gini", "bootstrap": "True"},
       "XGB": {"n_estimators": "300", "max_depth": "5", "learning_rate": "0.3", "subsample": "1.0", "colsample_bytree": "1.0"},
       "SVM": {"C": "5", "gamma": "scale", "kernel": "rbf"}, "MLP": {"hidden_layer_sizes": "(64, 32)", "alpha": "0.0001", "learning_rate_init": "0.001", "activation": "relu"}, "LR": {"C": "1"}}
YR = {"RF": (60, 84), "XGB": (76, 82), "SVM": (50, 82), "MLP": (74, 81), "LR": (55, 76)}
def hpanel(models, name, rows, cols):
    P = [(m, p) for m in models for p in hd[hd.m == m].p.unique()]
    fig, axs = plt.subplots(rows, cols, figsize=(W2, 1.45 * rows)); axs = axs.ravel()
    for ax, (m, p) in zip(axs, P):
        g = hd[(hd.m == m) & (hd.p == p)]; vals = list(dict.fromkeys(g.v)); mu = [g[g.v == v].acc.mean() * 100 for v in vals]; sg = [g[g.v == v].acc.std(ddof=0) * 100 for v in vals]
        ax.errorbar(range(len(vals)), mu, yerr=sg, color=COL[m], marker=MK[m], capsize=1.5, elinewidth=0.5)
        di = vals.index(DEF[m][p]); ax.plot(di, mu[di], marker="o", mfc="none", mec="#B02A2A", ms=6, mew=0.8)
        lbl = [v.replace("0.0001", "1e-4").replace("0.00001", "1e-5").replace("0.001", "1e-3").replace("0.01", "1e-2").replace("(", "").replace(")", "").replace(", ", "-").replace("hidden_layer_sizes", "hid") for v in vals]
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(lbl, rotation=45, ha="right", fontsize=5); ax.set_ylim(*YR[m]); ax.set_title(f"{m}: {p.replace('_', chr(92)+'_') if False else p}", fontsize=6)
        if ax in axs[::cols]: ax.set_ylabel("Accuracy (%)")
    for ax in axs[len(P):]: ax.axis("off")
    fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, name)
hpanel(["RF", "XGB"], "fig_hp_trees", 3, 4); hpanel(["SVM", "MLP", "LR"], "fig_hp_other", 2, 4)
# sensitivity summary
sens = []
for (m, p), g in hd.groupby(["m", "p"]): mu = g.groupby("v")["acc"].mean() * 100; sens.append((m, p, mu.max() - mu.min()))
sens.sort(key=lambda x: x[2]); fig, ax = plt.subplots(figsize=(W1, 3.5))
ax.barh(range(len(sens)), [s[2] for s in sens], color=[COL[s[0]] for s in sens], height=0.7)
ax.set_yticks(range(len(sens))); ax.set_yticklabels([f"{s[0]}: {s[1]}" for s in sens], fontsize=5.2); ax.set_xlabel("Accuracy range over tested values (pp)")
for i, s in enumerate(sens): ax.text(s[2] + 0.3, i, f"{s[2]:.1f}", va="center", fontsize=5)
ax.set_xlim(0, 25); save(fig, "fig_sens"); json.dump([(a, b, float(c)) for a, b, c in sens], open(os.path.join(RES, "sens.json"), "w"))

# ---------- SHAP ----------
imp = json.load(open(os.path.join(RES, "extra.json")))["shap"]; fig, axs = plt.subplots(2, 2, figsize=(W1, 3.0))
for ax, n in zip(axs.ravel(), NOISE_TYPES):
    top = sorted(imp[n].items(), key=lambda x: -x[1])[:6][::-1]; ax.barh(range(6), [t[1] for t in top], color=["#D55E00" if not t[0].startswith("p_") else "#9aa5b8" for t in top], height=0.7)
    ax.set_yticks(range(6)); ax.set_yticklabels([short.get(t[0], t[0]) for t in top], fontsize=5.2); ax.set_title(n.replace("_", " "), fontsize=6); ax.tick_params(axis="x", labelsize=5)
fig.supxlabel("mean |SHAP value|", fontsize=6); fig.tight_layout(h_pad=0.5, w_pad=0.5); save(fig, "fig_shap")
print("figs done")

# ---------- diagnostics (needs results/analysis.json from code/analysis.py) ----------
AN = json.load(open(os.path.join(RES, "analysis.json")))
SHORT = ["Depol.", "Amp. damp.", "Phase damp.", "Readout"]
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.75))
for ax, key, ttl in zip(axs, ["low_p_2048", "high_p_2048"], [r"(a) $p\leq0.005$", r"(b) $p\geq0.02$"]):
    C = np.array(AN["confusion"][key]); Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{100 * Cn[i, j]:.0f}", ha="center", va="center", fontsize=6, color="white" if Cn[i, j] > .5 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(SHORT, rotation=45, ha="right"); ax.set_title(ttl)
    ax.set_yticklabels(SHORT if ax is axs[0] else []); ax.set_xlabel("Predicted")
axs[0].set_ylabel("True"); fig.tight_layout(pad=0.3); save(fig, "fig_conf")

fig, axs = plt.subplots(1, 2, figsize=(W2, 2.0))
sev = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
for s, c in zip(["200", "2048", "32768"], ["#D55E00", "#0B1E3D", "#009E73"]):
    axs[0].plot(sev, [AN["mean_offsupport_shots"][s][str(p)] for p in sev], marker="o", color=c, label=f"{int(s):,} shots")
axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("noise severity $p$")
axs[0].set_ylabel("mean off-support shots $N$"); axs[0].set_title("(a) amount of noise signal"); axs[0].legend()
B = AN["acc_vs_offsupport_events"]; lab = [f"{r['lo']}" if r["lo"] == r["hi"] else (f"{r['lo']}-{r['hi']}" if r["hi"] else f">{r['lo'] - 1}") for r in B]
axs[1].bar(range(len(B)), [100 * r["acc"] for r in B], color="#1F3A6E"); axs[1].set_xticks(range(len(B))); axs[1].set_xticklabels(lab, rotation=45, ha="right")
axs[1].set_ylim(0, 110); axs[1].set_xlabel("number of off-support shots $N$ in the experiment")
axs[1].set_ylabel("RF accuracy (%)"); axs[1].set_title("(b) accuracy versus $N$ (non-phase-damping rows)")
for i, r in enumerate(B): axs[1].text(i, 100 * r["acc"] + 1.0, f"{100 * r['acc']:.0f}", ha="center", fontsize=5.5)
fig.tight_layout(pad=0.4); save(fig, "fig_events")

fig, axs = plt.subplots(1, 2, figsize=(W1, 1.75))
CS = ["Bell", "GHZ", "W", "GHZ4", "2xBell"]
for ax, kind, ttl in zip(axs, ["eng", "raw"], ["(a) engineered (6)", "(b) raw probabilities (28)"]):
    M = np.array(AN["transfer_train_row_test_col"][kind]) * 100; Mm = np.ma.masked_where(np.eye(5, dtype=bool), M)
    ax.imshow(Mm, cmap="Blues", vmin=20, vmax=80); ax.grid(False)
    for i in range(5):
        for j in range(5):
            if i != j: ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(CS, rotation=45, ha="right"); ax.set_yticklabels(CS if ax is axs[0] else []); ax.set_title(ttl); ax.set_xlabel("test circuit")
axs[0].set_ylabel("train circuit"); fig.tight_layout(pad=0.3); save(fig, "fig_transfer")
print("diagnostic figs done")
