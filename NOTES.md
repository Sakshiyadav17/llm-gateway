# Project notes (LLM Gateway)

## Data
- 939 questions: 400 GSM8K, 400 TriviaQA, 100 MMLU, 25 business, 15 Hinglish business
- Small model: Qwen2.5 1.5B (local). Big model: GPT-OSS 120B (Groq)
- Accuracy: small 55.6%, big 83.8% (oracle router 89.6%)
- Labels: exact match for GSM8K/TriviaQA/MMLU; business rows judged by the big model (biased, small n)

## Router (cascade, 5-fold CV on train+val, test set untouched)
- Baseline TF-IDF router: AUROC ~0.67
- With answer features + token confidence: LogReg AUROC 0.774, GradBoost 0.765
- Chosen: LogReg, threshold 0.69 (3-point tolerance) -> about 39% fewer big-model calls
- Trade-off table: data/tradeoff_table.csv

## PII shield (regex, no NER)
- Types: Aadhaar, PAN, phone, email, IFSC, GST
- Synthetic test 150/150 (self-generated, optimistic), hard cases 13/13 (written after seeing failures)
- Aadhaar: any 12-digit group starting 2-9 is masked (no checksum), by choice: higher recall, more false positives
- Lowercase PAN accepted, so some product codes may be masked
- Not covered: names, addresses, locations

## Limitations
- Small dataset and small test set; business labels judged by the big model
- Cascade design: small model runs first, so escalated queries pay both latencies
- "Calls saved" is not "cost saved" until the gateway computes token costs