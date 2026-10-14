"""Diagnostic analyses that explain the main results (uses the existing datasets; no new simulation).
Usage: python code/analysis.py     -> writes results/analysis.json
  A  confusion matrices (pooled RF, 5-fold CV)
  B  off-support events: how many shots actually carry the noise signal, and accuracy when there are none
  C  physics check: feature means per channel, and the analytic readout-error prediction for Bell
  D  train-on-one-circuit / test-on-another transfer matrix for three feature sets
  E  accuracy gain per doubling of shots
  F  accuracy versus number of off-support shots, pooled over all 12 shot settings
  G  mean unexpected-bitstring rate U by channel and severity (32768 shots)"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from expcore import make_model, yenc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

out = {}
def cvpred(df, seed=0):
    X, y = df[feat_cols(df, "all")].values, yenc(df)
    return cross_val_predict(make_model("RF"), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed)), y

# ---- A + B: confusion matrices and off-support analysis -------------------------------------------
conf, zero_frac, nevents, acc_u = {}, {}, {}, {}
for S in [200, 2048, 32768]:
    dfs = [load(S, d) for d in range(3)]
    preds = [cvpred(df, d) for d, df in enumerate(dfs)]
    big = pd.concat(dfs, ignore_index=True)
    big["pred"] = np.concatenate([p for p, _ in preds]); big["y"] = np.concatenate([y for _, y in preds])
    big["ok"] = big.pred == big.y
    conf[f"all_{S}"] = confusion_matrix(big.y, big.pred, labels=range(4)).tolist()
    if S == 2048:
        for nm, m in [("low_p", big.severity <= 0.005), ("high_p", big.severity >= 0.02)]:
            conf[f"{nm}_{S}"] = confusion_matrix(big.y[m], big.pred[m], labels=range(4)).tolist()
            out[f"acc_{nm}_{S}"] = float(big.ok[m].mean())
    # rows with no off-support shot at all (U == 0), excluding phase damping, which produces almost none
    nz = big[big.noise_type != "phase_damping"]
    zero_frac[S] = {str(p): float((g.unexpected_bitstring_rate == 0).mean()) for p, g in nz.groupby("severity")}
    nevents[S] = {str(p): float((g.unexpected_bitstring_rate * S).mean()) for p, g in nz.groupby("severity")}
    z = nz.unexpected_bitstring_rate == 0
    acc_u[S] = {"acc_U0": float(nz.ok[z].mean()) if z.any() else None, "n_U0": int(z.sum()),
                "acc_Upos": float(nz.ok[~z].mean()), "n_Upos": int((~z).sum()),
                "frac_U0_predicted_phase": float((nz.pred[z] == 2).mean()) if z.any() else None}
out.update(confusion=conf, zero_frac=zero_frac, mean_offsupport_shots=nevents, acc_by_U=acc_u)

# ---- C: physics check ------------------------------------------------------------------------------
big = pd.concat([load(32768, d) for d in range(3)], ignore_index=True)
hi = big[big.severity == 0.03]
out["feature_means_p003_32768"] = {nt: {f: float(hi[hi.noise_type == nt][f].mean()) for f in ENG} for nt in NOISE_TYPES}
b = pd.concat([load(2048, d) for d in range(3)], ignore_index=True)
b = b[(b.circuit == "bell") & (b.noise_type == "readout_error")]
out["bell_readout_U"] = {str(p): {"measured": float(g.unexpected_bitstring_rate.mean()), "analytic_2p(1-p)": 2 * p * (1 - p)}
                         for p, g in b.groupby("severity")}

# ---- D: transfer matrix ----------------------------------------------------------------------------
trans = {}
for kind in ["eng", "raw", "all"]:
    M = np.zeros((5, 5))
    for d in range(3):
        df = load(2048, d); cols = feat_cols(df, kind)
        for i, a in enumerate(CIRC_ORDER):
            tr = df[df.circuit == a]; clf = make_model("RF", n_jobs=4).fit(tr[cols].values, yenc(tr))
            for j, c in enumerate(CIRC_ORDER):
                te = df[df.circuit == c]
                M[i, j] += (clf.predict(te[cols].values) == yenc(te)).mean() / 3 if i != j else 0
    trans[kind] = M.tolist()
out["transfer_train_row_test_col"] = trans

# ---- E: gain per doubling of shots (pooled RF means from results/shots.jsonl) ---------------------
rows = [json.loads(l) for l in open(os.path.join(RES, "shots.jsonl"))]
acc = {}
for r in rows:
    s, d, sd, mod = r["key"].split("|")
    if mod == "RF": acc.setdefault(int(s), []).append(r["acc"])
S_ = sorted(acc); A = [100 * np.mean(acc[s]) for s in S_]; x = np.log2(S_)
def slope(lo, hi):
    m = [(lo <= s <= hi) for s in S_]; return float(np.polyfit(x[m], np.array(A)[m], 1)[0])
out["slope_points_per_doubling"] = {"50-2048": slope(50, 2048), "2048-32768": slope(2048, 32768), "all": slope(50, 32768)}

# ---- F: accuracy versus number of off-support shots ------------------------------------------------
edges = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 300), (301, 1000), (1001, 10**9)]
recs = []
for S in SHOTS_LIST:
    for d in range(3):
        df = load(S, d); pr, y = cvpred(df, d)
        recs.append(pd.DataFrame({"N": np.rint(df.unexpected_bitstring_rate * S).astype(int), "ok": pr == y,
                                  "nt": df.noise_type.values, "pred": pr}))
R = pd.concat(recs, ignore_index=True); R = R[R.nt != "phase_damping"]
out["acc_vs_offsupport_events"] = [{"lo": lo, "hi": (hi if hi < 10**9 else None), "n": int(((R.N >= lo) & (R.N <= hi)).sum()),
    "acc": float(R.ok[(R.N >= lo) & (R.N <= hi)].mean())} for lo, hi in edges if ((R.N >= lo) & (R.N <= hi)).any()]

# ---- G: mean U by channel and severity -------------------------------------------------------------
big = pd.concat([load(32768, d) for d in range(3)], ignore_index=True)
out["U_by_channel_severity_32768"] = {nt: {str(p): float(g.unexpected_bitstring_rate.mean()) for p, g in big[big.noise_type == nt].groupby("severity")} for nt in NOISE_TYPES}
json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1)
print("analysis done")
