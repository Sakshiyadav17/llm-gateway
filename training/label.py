"""Label each answer as correct/incorrect -> data/labeled.csv"""
import json
import re
import string
import time
import pandas as pd
from tqdm import tqdm
from gateway.models import ask_big


def norm(s):
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def to_num(s):
    s = str(s).replace(",", "").replace("$", "")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[-1].rstrip(".")) if nums else None


def check_gsm(ans, gold):
    try:
        g = float(gold)
    except ValueError:
        return False
    m = re.search(r"final answer:\s*(.*)", str(ans), re.IGNORECASE)
    text = (m.group(1) if m else str(ans)).replace(",", "")
    vals = [float(n.rstrip(".")) for n in re.findall(r"-?\d+\.?\d*", text)]
    if not vals:
        return False
    if m:  # accept if the first or last number in the final-answer line matches
        return abs(vals[0] - g) < 1e-6 or abs(vals[-1] - g) < 1e-6
    return abs(vals[-1] - g) < 1e-6  # no "Final answer" line: use the last number


def check_trivia(ans, gold, aliases):
    targets = {norm(gold)}
    if isinstance(aliases, str) and aliases:
        targets |= {norm(a) for a in json.loads(aliases)}
    padded = f" {norm(ans)} "
    return any(t and f" {t} " in padded for t in targets)


def check_mmlu(ans, gold):
    text = str(ans).strip().replace("*", "")
    m = re.match(r"^\(?([ABCD])\b", text)
    if not m:
        m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\b", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b([ABCD])\b", text)
    return bool(m) and m.group(1).upper() == gold


JUDGE = """You are grading an answer to a question.
Question: {q}
Reference answer: {gold}
Candidate answer: {ans}

Is the candidate answer correct, meaning it agrees with the reference in meaning
(extra detail is fine, wrong or contradictory facts are not)?
Reply with exactly one word: CORRECT or INCORRECT."""


def judge(q, gold, ans):
    for attempt in range(5):
        res = ask_big(JUDGE.format(q=q, gold=gold, ans=ans))
        text = (res["answer"] or "").upper()
        if not res["error"] and ("CORRECT" in text):
            return "INCORRECT" not in text
        time.sleep(3 * (attempt + 1))
    return None

def grade(row, which):
    ans = row[f"{which}_answer"]
    if not isinstance(ans, str) or not ans.strip():
        return None
    src = row["source"]
    if src == "gsm8k":
        return check_gsm(ans, row["gold"])
    if src == "triviaqa":
        return check_trivia(ans, row["gold"], row["aliases"])
    if src == "mmlu":
        return check_mmlu(ans, str(row["gold"]).strip())
    return judge(row["question"], row["gold"], ans)  # business questions


def main():
    q = pd.read_csv("data/questions.csv")
    a = pd.read_json("data/raw_answers.jsonl", lines=True).drop(columns=["source"])
    df = q.merge(a, on="id", how="inner")
    df = df[df["small_error"].isna() & df["big_error"].isna()].copy()
    print("Rows to label:", len(df))

    small, big = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        small.append(grade(row, "small"))
        big.append(grade(row, "big"))
    df["small_correct"] = small
    df["big_correct"] = big

    missing = int(df[["small_correct", "big_correct"]].isna().any(axis=1).sum())
    print("Rows with a missing grade (will be dropped):", missing)

    df = df.dropna(subset=["small_correct", "big_correct"])
    df["small_correct"] = df["small_correct"].astype(int)
    df["big_correct"] = df["big_correct"].astype(int)

    keep = ["id", "source", "question", "gold", "small_answer", "big_answer",
            "small_correct", "big_correct", "small_latency", "big_latency", "big_tokens"]
    df[keep].to_csv("data/labeled.csv", index=False)

    print("\nSaved", len(df), "rows to data/labeled.csv")
    print(df.groupby("source")[["small_correct", "big_correct"]].mean().round(3))
    print("\nOverall:", df[["small_correct", "big_correct"]].mean().round(3).to_dict())
    print("Small wrong, big right:", int(((df.small_correct == 0) & (df.big_correct == 1)).sum()))

if __name__ == "__main__":
    main()