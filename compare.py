"""Send questions to both models and compare."""
import sys
from gateway.models import ask_small, ask_big

DEFAULT_QUESTIONS = [
    "What is the capital of Australia?",
    "If a shirt costs 800 rupees and is discounted by 25%, what is the final price?",
    "Explain what a database index is in two sentences.",
]


def show(label, result):
    print(f"\n--- {label} ---")
    if result["error"]:
        print("ERROR:", result["error"])
    else:
        print(result["answer"].strip() or "(empty answer)")
        print(f"(latency: {result['latency']:.2f}s)")


if __name__ == "__main__":
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    for q in questions:
        print("\n" + "=" * 60)
        print("QUESTION:", q)
        show("SMALL (local)", ask_small(q))
        show("BIG (API)", ask_big(q))