import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

MODELS = ["RF", "XGB", "SVM", "MLP", "LR"]
def make_model(name, **p):
    if name == "RF":  return RandomForestClassifier(**{**dict(n_estimators=300, random_state=0, n_jobs=1), **p})
    if name == "XGB": return xgb.XGBClassifier(**{**dict(n_estimators=300, max_depth=5, random_state=0, eval_metric="mlogloss", n_jobs=1, tree_method="hist"), **p})
    if name == "SVM": return make_pipeline(StandardScaler(), SVC(**{**dict(kernel="rbf", C=5, gamma="scale"), **p}))
    if name == "MLP": return make_pipeline(StandardScaler(), MLPClassifier(**{**dict(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0), **p}))
    if name == "LR":  return make_pipeline(StandardScaler(), LogisticRegression(**{**dict(max_iter=1000), **p}))
def yenc(df): return df["noise_type"].map({n: i for i, n in enumerate(NOISE_TYPES)}).values

def cv_eval(df, model, params=None, seed=0, k=5, feats="all", detail=False):
    cols = feats if isinstance(feats, list) else feat_cols(df, feats)
    X, y = df[cols].values, yenc(df)
    pred = cross_val_predict(make_model(model, **(params or {})), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed))
    ok = (pred == y); out = {"acc": float(ok.mean())}
    if detail:
        d = df.assign(_ok=ok)
        out["by_sev"] = {str(k_): float(v) for k_, v in d.groupby("severity")["_ok"].mean().items()}
        out["by_circ"] = {k_: float(v) for k_, v in d.groupby("circuit")["_ok"].mean().items()}
        out["by_noise"] = {k_: float(v) for k_, v in d.groupby("noise_type")["_ok"].mean().items()}
    return out
