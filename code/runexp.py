import sys, time, json, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expcore import *

DR = [0, 1, 2]
HP = {
 "RF": {"n_estimators": [10, 25, 50, 100, 200, 300, 500], "max_depth": [2, 4, 6, 8, 12, None],
        "max_features": ["sqrt", "log2", 0.5, 1.0], "min_samples_split": [2, 4, 8, 16],
        "min_samples_leaf": [1, 2, 4, 8], "criterion": ["gini", "entropy"], "bootstrap": [True, False]},
 "XGB": {"n_estimators": [50, 100, 200, 300, 500], "max_depth": [2, 3, 5, 7, 9],
         "learning_rate": [0.01, 0.03, 0.1, 0.3, 0.5], "subsample": [0.5, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0]},
 "SVM": {"C": [0.1, 1, 5, 10, 50, 100], "gamma": ["scale", 0.001, 0.01, 0.1, 1], "kernel": ["linear", "poly", "rbf", "sigmoid"]},
 "MLP": {"hidden_layer_sizes": [(16,), (32,), (64,), (64, 32), (128, 64), (128, 64, 32)], "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
         "learning_rate_init": [1e-4, 1e-3, 1e-2], "activation": ["relu", "tanh", "logistic"]},
 "LR": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
}
GRID = {
 "RF": {"n_estimators": [100, 300], "max_depth": [8, None], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 2]},
 "XGB": {"max_depth": [3, 5, 7], "learning_rate": [0.1, 0.3], "n_estimators": [100, 300]},
 "SVM": {"C": [1, 5, 10, 50], "gamma": ["scale", 0.01, 0.1]},
 "MLP": {"hidden_layer_sizes": [(64,), (64, 32), (128, 64)], "alpha": [1e-4, 1e-2]},
 "LR": {"C": [0.01, 0.1, 1, 10, 100]},
}
POOL = None
def pool():
    global POOL
    if POOL is None: POOL = pd.concat([load(2048, d).assign(draw=d) for d in DR], ignore_index=True)
    return POOL
def strat_sub(df, n, seed):
    per = n // 20; parts = [g.sample(per, random_state=seed) for _, g in df.groupby(["circuit", "noise_type"])]
    return pd.concat(parts).reset_index(drop=True)

def tasks(exp):
    T = []
    if exp == "shots":
        for s in SHOTS_LIST:
            for d in DR:
                for sd in [0, 1]:
                    for m in MODELS:
                        T.append((f"{s}|d{d}|s{sd}|{m}", lambda s=s, d=d, sd=sd, m=m: cv_eval(load(s, d), m, seed=sd, detail=True)))
    if exp == "size":
        for rep in range(3):
            for n in [100, 200, 300, 400, 500, 600, 720, 1080, 1440, 2160]:
                for m in MODELS:
                    T.append((f"{n}|r{rep}|{m}", lambda n=n, rep=rep, m=m: cv_eval(strat_sub(pool(), n, rep) if n < 2160 else pool(), m, seed=rep)))
    if exp == "feat":
        settings = {"all": None, "eng": ENG}
        D0 = load(2048, 0); raw = feat_cols(D0, "raw")
        settings["raw"] = raw
        for f in ENG: settings[f"drop_{f}"] = [c for c in ENG + raw if c != f]
        for f in ENG: settings[f"only_{f}"] = [f]
        for name, cols in settings.items():
            for d in DR:
                for sd in [0, 1]:
                    T.append((f"{name}|d{d}|s{sd}", lambda name=name, cols=cols, d=d, sd=sd: cv_eval(load(2048, d), "RF", seed=sd, feats=(cols if cols else "all"))))
    if exp == "sev":
        for th in SEVERITIES[:-1]:
            for d in DR:
                for sd in [0, 1]:
                    T.append((f"ge{th}|d{d}|s{sd}", lambda th=th, d=d, sd=sd: cv_eval(load(2048, d).query("severity >= @th").reset_index(drop=True), "RF", seed=sd)))
    if exp == "circ":
        for npool in range(1, 6):
            for d in DR:
                for sd in [0, 1]:
                    T.append((f"pool{npool}|d{d}|s{sd}", lambda npool=npool, d=d, sd=sd: cv_eval(load(2048, d)[lambda x: x.circuit.isin(CIRC_ORDER[:npool])].reset_index(drop=True), "RF", seed=sd)))
        def loco(c, kind, d):
            df = load(2048, d); cols = feat_cols(df, kind)
            tr, te = df[df.circuit != c], df[df.circuit == c]
            m = make_model("RF").fit(tr[cols].values, yenc(tr)); return {"acc": float((m.predict(te[cols].values) == yenc(te)).mean())}
        for c in CIRC_ORDER:
            for kind in ["all", "eng"]:
                for d in DR:
                    T.append((f"loco|{c}|{kind}|d{d}", lambda c=c, kind=kind, d=d: loco(c, kind, d)))
    if exp == "hp":
        for m, ps in HP.items():
            for p, vals in ps.items():
                for v in vals:
                    for d in DR:
                        T.append((f"{m}|{p}|{v}|d{d}", lambda m=m, p=p, v=v, d=d: cv_eval(load(2048, d), m, params={p: v}, seed=0)))
    if exp == "cv":
        for k in [3, 5, 10]:
            for m in MODELS:
                for d in DR:
                    T.append((f"k{k}|{m}|d{d}", lambda k=k, m=m, d=d: cv_eval(load(2048, d), m, seed=0, k=k)))
    if exp == "main":
        for m in MODELS:
            for c in CIRC_ORDER:
                for d in DR:
                    for sd in [0, 1]:
                        T.append((f"{m}|{c}|d{d}|s{sd}", lambda m=m, c=c, d=d, sd=sd: cv_eval(load(2048, d).query("circuit == @c").reset_index(drop=True), m, seed=sd)))
    if exp == "tune":
        for m, g in GRID.items():
            keys = list(g)
            for combo in product(*[g[k] for k in keys]):
                params = dict(zip(keys, combo))
                for d, sds in [(0, [0]), (1, [0]), (2, [0, 1, 2])]:
                    for sd in sds:
                        T.append((f"{m}|{json.dumps(params, default=str)}|d{d}|s{sd}", lambda m=m, params=params, d=d, sd=sd: cv_eval(load(2048, d), m, params=params, seed=sd)))
    return T

if __name__ == "__main__":
    exp, budget = sys.argv[1], float(sys.argv[2])
    f = f"{RES}/{exp}.jsonl"; done = set()
    if os.path.exists(f):
        for l in open(f): done.add(json.loads(l)["key"])
    T = [t for t in tasks(exp) if t[0] not in done]; total = len(done) + len(T)
    t0 = time.time(); n = 0
    with open(f, "a") as out:
        for key, fn in T:
            if time.time() - t0 > budget: break
            r = fn(); r["key"] = key; out.write(json.dumps(r) + "\n"); out.flush(); n += 1
    print(f"{exp}: ran {n} this call; {len(done)+n}/{total} done ({time.time()-t0:.0f}s)")
