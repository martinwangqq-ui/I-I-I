from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import os
import json

app = Flask(__name__, static_folder=".", static_url_path="")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "You are a precise micro-poetry generator. "
    "Given the current sentence that starts with 'I', "
    "propose 3-5 SINGLE-WORD English candidates that can follow naturally. "
    "They must be grammatical, readable, poetic, and from the allowed POS tags. "
    "Return JSON only."
)

@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/next")
def next_word():
    data = request.get_json()
    sentence   = data.get("sentence", "I")
    allowedpos = data.get("allowed_pos", ["verb", "adjective", "adverb", "noun"])
    k          = int(data.get("k", 4))

    user_prompt = f"""
Current sentence: {sentence}
Allowed POS: {allowedpos}
You must respond with ONLY valid JSON, no extra text.

Return strictly this JSON schema:
{{
  "options": [{{"word": "string", "pos": "verb|adjective|adverb|noun|conjunction"}}],
  "soft_fix": "string"
}}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.7,
        top_p=0.9,
    )

    try:
        try:
            text = resp.output[0].content[0].text
        except Exception:
            text = str(resp)

        data = json.loads(text)
    except Exception as e:
        print("Failed to return JSON：", e)
        print("Original contex：", text)
        data = {"options": [], "soft_fix": ""}

    return jsonify(data)


if __name__ == "__main__":
    print("Flask server running at http://127.0.0.1:5000")
    print("API KEY =", os.getenv("OPENAI_API_KEY"))

    app.run(debug=True)
