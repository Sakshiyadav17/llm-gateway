"""Improved router. Uses train+val only unless --final is passed."""
import os, sys, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 42
FINAL = "--final" in sys.argv
TOL = 0.03
HEDGE = ["not sure", "unknown", "cannot", "can't", "don't know", "unclear", "no information"]

df = pd.read_csv("data/labeled.csv")
sp = pd.read_csv("data/splits.csv")
df = df.merge(sp, on="id")
USE_CONF = os.path.exists("data/confidence.jsonl")
CONF = ["mean_lp", "min_lp", "first_lp", "frac_low", "n_tokens"]
if USE_CONF:
    c = pd.read_json("data/confidence.jsonl", lines=True)
    df = df.merge(c, on="id", how="inner")
print("Using confidence features:", USE_CONF, "| rows:", len(df))


def hand(d):
    q, a = d["question"], d["small_answer"].fillna("")
    f = pd.DataFrame({
        "q_len": np.log1p(q.str.len()),
        "q_words": np.log1p(q.str.split().str.len()),
        "q_digits": np.log1p(q.str.count(r"\d")),
        "has_options": q.str.contains(r"\nA\. ").astype(int),
        "wh_word": q.str.lower().str.match(r"^(who|what|which|where|when|whom|whose)\b").astype(int),
        "is_math_like": q.str.contains(r"how (?:many|much)|total|each|per ", case=False).astype(int),
        "caps_ratio": q.apply(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1)),
        "a_len": np.log1p(a.str.len()),
        "a_hedge": a.str.lower().apply(lambda s: sum(h in s for h in HEDGE)),
        "a_has_final": a.str.contains("final answer", case=False).astype(int),
        "a_is_short": (a.str.len() < 25).astype(int),
    })
    if USE_CONF:
        for k in CONF:
            f[k] = d[k].values
    return f.values.astype(float)


train, val, test = (df[df.split == s] for s in ["train", "val", "test"])
print(f"train={len(train)} val={len(val)} test={len(test)}")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
svd = TruncatedSVD(60, random_state=SEED).fit(vec.fit_transform(train["question"]))
scaler = StandardScaler().fit(hand(train))


def X(d):
    return np.hstack([svd.transform(vec.transform(d["question"])), hand(d)])


def Xs(d):
    return np.hstack([svd.transform(vec.transform(d["question"])), scaler.transform(hand(d))])


y = train["small_correct"].values
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                    l2_regularization=1.0, random_state=SEED).fit(X(train), y)
lr = LogisticRegression(C=0.5, max_iter=3000).fit(Xs(train), y)
models = {"LogReg+feats": lambda d: lr.predict_proba(Xs(d))[:, 1],
          "GradBoost+feats": lambda d: gb.predict_proba(X(d))[:, 1]}

TH = np.linspace(0, 1, 101)


def curve(p, s, b):
    return pd.DataFrame([(t, np.where(p >= t, s, b).mean(), 1 - (p >= t).mean()) for t in TH],
                        columns=["threshold", "accuracy", "pct_big"])


def pick(c, big_acc):
    ok = c[c.accuracy >= big_acc - TOL]
    return float(ok.sort_values("pct_big").iloc[0].threshold) if len(ok) else 0.0


vs, vb = val.small_correct.values, val.big_correct.values
chosen = {}
for n, f in models.items():
    p = f(val)
    chosen[n] = pick(curve(p, vs, vb), vb.mean())
    print(f"[val] {n}: AUROC={roc_auc_score(vs, p):.3f} threshold={chosen[n]:.2f}")

ev, nm = (test, "TEST") if FINAL else (val, "VALIDATION")
s, b = ev.small_correct.values, ev.big_correct.values
print(f"\n=== {nm} results (n={len(ev)}) ===")
print(f"Always small: {s.mean():.3f} | Always big: {b.mean():.3f} | Oracle: {np.maximum(s, b).mean():.3f}")
plt.figure(figsize=(7, 5))
for n, f in models.items():
    p = f(ev)
    c = curve(p, s, b)
    pt = c.iloc[(c.threshold - chosen[n]).abs().argmin()]
    print(f"{n}: AUROC={roc_auc_score(s, p):.3f} | threshold {chosen[n]:.2f} -> "
          f"accuracy={pt.accuracy:.3f}, big usage={pt.pct_big:.1%}")
    plt.plot(c.pct_big * 100, c.accuracy * 100, label=n)
    plt.scatter([pt.pct_big * 100], [pt.accuracy * 100], zorder=5)
plt.scatter([0, 100], [s.mean() * 100, b.mean() * 100], c="black", marker="s", label="Always small / big")
plt.xlabel("% of queries sent to big model"); plt.ylabel("Accuracy (%)")
plt.title(f"Improved router: cost vs accuracy ({nm.lower()})"); plt.legend(); plt.grid(alpha=0.3)
out = "data/cost_accuracy_v2_test.png" if FINAL else "data/cost_accuracy_v2_val.png"
plt.savefig(out, dpi=150, bbox_inches="tight"); print("Saved", out)
if not FINAL:
    joblib.dump({"vec": vec, "svd": svd, "scaler": scaler, "lr": lr, "gb": gb,
                 "thresholds": chosen, "use_conf": USE_CONF}, "data/router_v2.joblib")