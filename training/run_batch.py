"""Run every question through both models. Resumable: rerun it and it continues."""
import json
import os
import sys
import time
import pandas as pd
from tqdm import tqdm
from gateway.models import ask_small, ask_big

OUT = "data/raw_answers.jsonl"

SUFFIX = {
    "gsm8k": "\n\nSolve step by step. End with a final line exactly like: Final answer: <number>",
    "triviaqa": "\n\nAnswer in a few words only.",
    "mmlu": "",
    "business": "\n\nAnswer concisely.",
    "business_hinglish": "\n\nAnswer concisely.",
}


def with_retry(fn, prompt, tries=5):
    res = None
    for t in range(tries):
        res = fn(prompt)
        if not res["error"] and (res["answer"] or "").strip():
            return res
        err = res["error"] or "empty answer"
        wait = 20 * (t + 1) if "429" in err else 5 * (t + 1)
        time.sleep(wait)
    return res


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = pd.read_csv("data/questions.csv").sample(frac=1, random_state=42)  # shuffled
    if limit:
        df = df.head(limit)

    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = df[~df["id"].isin(done)]
    print(f"{len(done)} already done, {len(todo)} to go")

    with open(OUT, "a", encoding="utf-8") as f:
        for _, row in tqdm(todo.iterrows(), total=len(todo)):
            prompt = row["question"] + SUFFIX.get(row["source"], "")
            s = with_retry(ask_small, prompt)
            b = with_retry(ask_big, prompt)
            rec = {
                "id": row["id"], "source": row["source"],
                "small_answer": s["answer"], "small_latency": s["latency"], "small_error": s["error"],
                "big_answer": b["answer"], "big_latency": b["latency"],
                "big_tokens": b.get("tokens"), "big_error": b["error"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()