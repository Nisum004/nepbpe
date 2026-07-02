import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("groq_api"))

categories = [
    "नेपालको भूगोल र प्रकृति",
    "नेपालको इतिहास र संस्कृति",
    "नेपालको राजनीति र सरकार",
    "नेपाली भाषा र साहित्य",
    "विज्ञान र प्रविधि",
    "गणित र तर्कशक्ति",
    "स्वास्थ्य र जीवनशैली",
    "शिक्षा र करियर",
    "खेलकुद र मनोरञ्जन",
    "दैनिक जीवन र कुराकानी",
    "नेपालका चाडपर्व र धर्म",
    "नेपालको अर्थतन्त्र र व्यापार",
    "कम्प्युटर विज्ञान र AI",
    "नेपालका नदी र हिमाल",
    "नेपालको खानपान र संस्कृति",
]

all_pairs = []

for category in categories:
    print(f"\n📝 Generating for: {category}...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # best free groq model
            messages=[{
                "role": "user",
                "content": f"""Generate 30 Nepali question-answer pairs about the topic: '{category}'.

Format your response as a JSON array ONLY like this:
[
  {{"q": "नेपाली प्रश्न?", "a": "नेपाली उत्तर।"}},
  {{"q": "अर्को प्रश्न?", "a": "अर्को उत्तर।"}}
]

Strict rules:
- Questions and answers MUST be in Nepali Devanagari script only
- No English words at all
- Answers should be 1-3 sentences, clear and factual
- Questions should sound natural and conversational
- Return ONLY the JSON array, no explanation, no markdown, no backticks"""
            }],
            temperature=0.7,
            max_tokens=6000,
        )

        text = response.choices[0].message.content.strip()
        # Clean any markdown if present
        text = text.replace("```json", "").replace("```", "").strip()

        # Find JSON array in response
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > start:
            text = text[start:end]

        pairs = json.loads(text)
        all_pairs.extend(pairs)
        print(f"  ✅ Got {len(pairs)} pairs — total so far: {len(all_pairs)}")

    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse failed: {e}")
        print(f"  Raw response: {text[:200]}")
    except Exception as e:
        print(f"  ❌ API error: {e}")

print(f"\n✅ Total pairs generated: {len(all_pairs)}")

# Combine with existing manual pairs
existing = []
if os.path.exists("data/chat_data.txt"):
    print("📂 Found existing chat_data.txt — will append to it")

os.makedirs("data", exist_ok=True)

# Save training text
with open("data/chat_data.txt", "a", encoding="utf-8") as f:
    for pair in all_pairs:
        q = pair.get("q", "").strip()
        a = pair.get("a", "").strip()
        if q and a:
            f.write(f"प्रश्न । {q}\nउत्तर । {a}\nअन्त्य ।\n\n")

# Save raw JSON for reference
with open("data/qa_pairs.json", "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, ensure_ascii=False, indent=2)

# Count final lines
with open("data/chat_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"💾 Training file: data/chat_data.txt")
print(f"💾 Raw JSON:      data/qa_pairs.json")
print(f"📊 Total Q&A pairs in training file: {len([l for l in lines if l.startswith('प्रश्न ।')])}")