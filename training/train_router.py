"""Train a router that predicts whether the small model will answer correctly.
Default run uses train + validation only. The test set is touched only with --final."""
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

SEED = 42
FINAL = "--final" in sys.argv
TOLERANCE = 0.03  # accept up to 3 points below always-big accuracy

df = pd.read_csv("data/labeled.csv")
trval, test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["source"])
train, val = train_test_split(trval, test_size=0.15 / 0.85, random_state=SEED,
                              stratify=trval["source"])
splits = pd.concat([train.assign(split="train"), val.assign(split="val"),
                    test.assign(split="test")])[["id", "split"]]
splits.to_csv("data/splits.csv", index=False)
print(f"train={len(train)} val={len(val)} test={len(test)}")


def length_feats(q):
    return np.c_[np.log1p(q.str.len()), np.log1p(q.str.count(r"\d")),
                 np.log1p(q.str.count("\n"))]


vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
T_train = vec.fit_transform(train["question"])
svd = TruncatedSVD(100, random_state=SEED).fit(T_train)


def feats_lr(d):
    return hstack([vec.transform(d["question"]), csr_matrix(length_feats(d["question"]))]).tocsr()


def feats_gb(d):
    return np.hstack([svd.transform(vec.transform(d["question"])), length_feats(d["question"])])


y_train = train["small_correct"].values
lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats_lr(train), y_train)
gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                    random_state=SEED).fit(feats_gb(train), y_train)
models = {"LogReg": (lambda d: lr.predict_proba(feats_lr(d))[:, 1]),
          "GradBoost": (lambda d: gb.predict_proba(feats_gb(d))[:, 1])}

THRESHOLDS = np.linspace(0, 1, 101)


def curve(p, small, big):
    rows = []
    for t in THRESHOLDS:
        to_small = p >= t
        acc = np.where(to_small, small, big).mean()
        rows.append((t, acc, 1 - to_small.mean()))
    return pd.DataFrame(rows, columns=["threshold", "accuracy", "pct_big"])


def pick_threshold(c, big_acc):
    ok = c[c["accuracy"] >= big_acc - TOLERANCE]
    return float(ok.sort_values("pct_big").iloc[0]["threshold"]) if len(ok) else 0.0


# Thresholds are always chosen on the VALIDATION set
vsmall, vbig = val["small_correct"].values, val["big_correct"].values
chosen = {}
for name, f in models.items():
    p = f(val)
    c = curve(p, vsmall, vbig)
    chosen[name] = pick_threshold(c, vbig.mean())
    print(f"[val] {name}: AUROC={roc_auc_score(vsmall, p):.3f} "
          f"threshold={chosen[name]:.2f}")

# Evaluate on val (default) or test (--final) using the validation thresholds
ev, ev_name = (test, "TEST") if FINAL else (val, "VALIDATION")
small, big = ev["small_correct"].values, ev["big_correct"].values
print(f"\n=== {ev_name} results (n={len(ev)}) ===")
print(f"Always small: accuracy={small.mean():.3f}, big usage=0%")
print(f"Always big:   accuracy={big.mean():.3f}, big usage=100%")
oracle = np.maximum(small, big).mean()
print(f"Oracle (perfect router): accuracy={oracle:.3f}")

plt.figure(figsize=(7, 5))
results = []
for name, f in models.items():
    p = f(ev)
    c = curve(p, small, big)
    plt.plot(c["pct_big"] * 100, c["accuracy"] * 100, label=name)
    t = chosen[name]
    pt = c.iloc[(c["threshold"] - t).abs().argmin()]
    print(f"{name}: AUROC={roc_auc_score(small, p):.3f} | threshold {t:.2f} -> "
          f"accuracy={pt.accuracy:.3f}, big usage={pt.pct_big:.1%}")
    results.append((name, t, float(pt.accuracy), float(pt.pct_big)))
    plt.scatter([pt.pct_big * 100], [pt.accuracy * 100], zorder=5)
plt.scatter([0, 100], [small.mean() * 100, big.mean() * 100], c="black", marker="s",
            label="Always small / always big")
plt.xlabel("% of queries sent to big model (proxy for cost)")
plt.ylabel("Accuracy (%)")
plt.title(f"Cost vs accuracy ({ev_name.lower()} set)")
plt.legend()
plt.grid(alpha=0.3)
out = "data/cost_accuracy_test.png" if FINAL else "data/cost_accuracy_val.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Saved plot:", out)

if not FINAL:
    joblib.dump({"vec": vec, "svd": svd, "lr": lr, "gb": gb, "thresholds": chosen},
                "data/router.joblib")
    json.dump(chosen, open("data/router_thresholds.json", "w"))
    print("Saved data/router.joblib")