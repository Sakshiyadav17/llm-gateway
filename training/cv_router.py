"""5-fold cross-validated router evaluation on train+val only (test stays locked).
Produces out-of-fold predictions, then a trade-off table across tolerances."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
HEDGE = ["not sure", "unknown", "cannot", "can't", "don't know", "unclear", "no information"]
CONF = ["mean_lp", "min_lp", "first_lp", "frac_low", "n_tokens"]

df = pd.read_csv("data/labeled.csv").merge(pd.read_csv("data/splits.csv"), on="id")
df = df.merge(pd.read_json("data/confidence.jsonl", lines=True), on="id", how="inner")
df = df[df.split != "test"].reset_index(drop=True)   # test set stays locked
print("Rows used (train+val only):", len(df))


def hand(d):
    q, a = d["question"], d["small_answer"].fillna("")
    f = pd.DataFrame({
        "q_len": np.log1p(q.str.len()),
        "q_words": np.log1p(q.str.split().str.len()),
        "q_digits": np.log1p(q.str.count(r"\d")),
        "has_options": q.str.contains(r"\nA\. ").astype(int),
        "wh_word": q.str.lower().str.match(r"^(?:who|what|which|where|when|whom|whose)\b").astype(int),
        "is_math_like": q.str.contains(r"how (?:many|much)|total|each|per ", case=False).astype(int),
        "caps_ratio": q.apply(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1)),
        "a_len": np.log1p(a.str.len()),
        "a_hedge": a.str.lower().apply(lambda s: sum(h in s for h in HEDGE)),
        "a_has_final": a.str.contains("final answer", case=False).astype(int),
        "a_is_short": (a.str.len() < 25).astype(int),
    })
    for k in CONF:
        f[k] = d[k].values
    return f.values.astype(float)


strat = df["source"] + "_" + df["small_correct"].astype(str)
oof = {"LogReg": np.zeros(len(df)), "GradBoost": np.zeros(len(df))}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for fold, (tr, te) in enumerate(skf.split(df, strat), 1):
    a, b = df.iloc[tr], df.iloc[te]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    svd = TruncatedSVD(60, random_state=SEED).fit(vec.fit_transform(a["question"]))
    sc = StandardScaler().fit(hand(a))

    def X(d):
        return np.hstack([svd.transform(vec.transform(d["question"])), hand(d)])

    def Xs(d):
        return np.hstack([svd.transform(vec.transform(d["question"])), sc.transform(hand(d))])

    y = a["small_correct"].values
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250,
                                        l2_regularization=1.0, random_state=SEED).fit(X(a), y)
    lr = LogisticRegression(C=0.5, max_iter=3000).fit(Xs(a), y)
    oof["GradBoost"][te] = gb.predict_proba(X(b))[:, 1]
    oof["LogReg"][te] = lr.predict_proba(Xs(b))[:, 1]
    print(f"fold {fold} done")

s, b = df["small_correct"].values, df["big_correct"].values
big_acc, small_acc, oracle = b.mean(), s.mean(), np.maximum(s, b).mean()
print(f"\nAlways small: {small_acc:.3f} | Always big: {big_acc:.3f} | Oracle: {oracle:.3f}")

TH = np.linspace(0, 1, 201)


def curve(p):
    return pd.DataFrame([(t, np.where(p >= t, s, b).mean(), 1 - (p >= t).mean()) for t in TH],
                        columns=["threshold", "accuracy", "pct_big"])


plt.figure(figsize=(7, 5))
rows, thresholds = [], {}
for name, p in oof.items():
    print(f"\n{name}: out-of-fold AUROC = {roc_auc_score(s, p):.3f}")
    c = curve(p)
    plt.plot(c.pct_big * 100, c.accuracy * 100, label=name)
    for tol in [0.02, 0.03, 0.05, 0.08]:
        ok = c[c.accuracy >= big_acc - tol].sort_values("pct_big")
        if len(ok):
            r = ok.iloc[0]
            rows.append({"router": name, "tolerance": f"{tol*100:.0f} pts",
                         "threshold": round(r.threshold, 3),
                         "accuracy": round(r.accuracy, 3),
                         "big_usage": f"{r.pct_big:.1%}",
                         "calls_saved": f"{1 - r.pct_big:.1%}"})
            thresholds[(name, tol)] = float(r.threshold)
plt.scatter([0, 100], [small_acc * 100, big_acc * 100], c="black", marker="s",
            label="Always small / big")
plt.plot([0, 100], [small_acc * 100, big_acc * 100], "k--", alpha=0.4, label="Random routing")
plt.xlabel("% of queries sent to big model"); plt.ylabel("Accuracy (%)")
plt.title("Cross-validated cost vs accuracy (train+val, 5-fold)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("data/cost_accuracy_cv.png", dpi=150, bbox_inches="tight")

res = pd.DataFrame(rows)
print("\n=== Trade-off table (out-of-fold) ===")
print(res.to_string(index=False))
res.to_csv("data/tradeoff_table.csv", index=False)
print("\nSaved data/cost_accuracy_cv.png and data/tradeoff_table.csv")