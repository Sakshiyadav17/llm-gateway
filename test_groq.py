import requests
from gateway.models import GROQ_API_KEY, GROQ_MODEL, GROQ_URL

print("Model:", repr(GROQ_MODEL))

r = requests.post(
    GROQ_URL,
    headers={"Authorization": "Bearer " + GROQ_API_KEY},
    json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}]},
    timeout=60,
)
print("Status:", r.status_code)
print(r.text)

print("\n--- Models available to your key ---")
m = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": "Bearer " + GROQ_API_KEY},
    timeout=60,
)
print("Status:", m.status_code)
if m.ok:
    for item in m.json()["data"]:
        print(item["id"])
else:
    print(m.text)