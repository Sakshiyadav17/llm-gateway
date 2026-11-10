"""Re-ask the small model every question and record token-probability features.
Resumable: rerun and it continues."""
import json, os, math
import pandas as pd
import requests
from tqdm import tqdm
from gateway.models import OLLAMA_URL, OLLAMA_MODEL
from training.run_batch import SUFFIX

OUT = "data/confidence.jsonl"


def get_lp(prompt):
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": OLLAMA_MODEL, "stream": False, "logprobs": True,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0}}, timeout=300)
    j = r.json()
    lps = j.get("logprobs") or j.get("message", {}).get("logprobs") or []
    return [t["logprob"] for t in lps if "logprob" in t]


df = pd.read_csv("data/labeled.csv")
q = pd.read_csv("data/questions.csv").set_index("id")
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()}
todo = df[~df["id"].isin(done)]
print(len(done), "done,", len(todo), "to go")

with open(OUT, "a", encoding="utf-8") as f:
    for _, row in tqdm(todo.iterrows(), total=len(todo)):
        prompt = row["question"] + SUFFIX.get(row["source"], "")
        try:
            lp = get_lp(prompt)
        except Exception:
            continue
        if not lp:
            continue
        probs = [math.exp(x) for x in lp]
        rec = {"id": row["id"], "mean_lp": sum(lp) / len(lp), "min_lp": min(lp),
               "first_lp": lp[0], "frac_low": sum(p < 0.5 for p in probs) / len(probs),
               "n_tokens": len(lp)}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        