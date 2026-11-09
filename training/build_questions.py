"""Build data/questions.csv from GSM8K, TriviaQA and MMLU."""
import json
import random
import pandas as pd
from datasets import load_dataset

SEED = 42
N_GSM, N_TRIVIA, N_MMLU = 400, 400, 100
random.seed(SEED)
rows = []


def sample(ds, n):
    idx = random.sample(range(len(ds)), n)
    return [ds[i] for i in idx]


# 1. GSM8K: math word problems. Gold answer is the number after "####".
print("Loading GSM8K...")
gsm = load_dataset("openai/gsm8k", "main", split="test")
for i, ex in enumerate(sample(gsm, N_GSM)):
    gold = ex["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({"id": f"gsm_{i}", "source": "gsm8k",
                 "question": ex["question"], "gold": gold, "aliases": ""})

# 2. TriviaQA: factual questions. Gold has several acceptable aliases.
print("Loading TriviaQA...")
tqa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
for i, ex in enumerate(sample(tqa, N_TRIVIA)):
    rows.append({"id": f"trivia_{i}", "source": "triviaqa",
                 "question": ex["question"], "gold": ex["answer"]["value"],
                 "aliases": json.dumps(ex["answer"]["aliases"])})

# 3. MMLU: multiple choice. Question includes options, gold is a letter.
print("Loading MMLU...")
mmlu = load_dataset("cais/mmlu", "all", split="test")
for i, ex in enumerate(sample(mmlu, N_MMLU)):
    letters = "ABCD"
    opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(ex["choices"]))
    q = f"{ex['question']}\n{opts}\nAnswer with only the letter (A, B, C or D)."
    rows.append({"id": f"mmlu_{i}", "source": "mmlu",
                 "question": q, "gold": letters[ex["answer"]], "aliases": ""})

df = pd.DataFrame(rows)
df.to_csv("data/questions.csv", index=False)
print("\nSaved", len(df), "questions to data/questions.csv")
print(df["source"].value_counts())