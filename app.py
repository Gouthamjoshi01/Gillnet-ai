#!/usr/bin/env python3
"""Gillnet-ai web UI backend. Run: python app.py -> http://127.0.0.1:5000"""
import os
import re

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from gillnet import MODEL_PATH, PhishModel

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="")

URL_RE = re.compile(r"""(?:https?://|www\.)[^\s<>"'\)\]]+""", re.I)

CUES = [
    r"urgent(ly)?", r"verify your (account|identity|email)", r"(account|password)[^.\n]{0,25}(suspend|expire|lock)",
    r"click (here|below|the link)", r"within (24|48) hours", r"unusual (activity|sign[- ]?in)",
    r"confirm your", r"limited time", r"update your (payment|billing)", r"act now", r"immediate action",
]
CUE_RE = [re.compile(c, re.I) for c in CUES]


def load_model():
    path = os.path.join(BASE, MODEL_PATH)
    if os.path.exists(path):
        return joblib.load(path)
    sample = os.path.join(BASE, "sample_data.csv")
    print("[!] No trained model found - training on sample_data.csv (demo mode).")
    df = pd.read_csv(sample)
    model = PhishModel().fit(df["url"].astype(str).tolist(), df["label"].astype(int).values)
    joblib.dump(model, path)
    return model


MODEL = load_model()


def extract_urls(text: str):
    urls = [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]
    if not urls and " " not in text.strip() and "." in text:
        urls = [text.strip()]
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:20]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/scan")
def scan():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="empty input"), 400
    urls = extract_urls(text)
    cues = sorted({m.group(0).lower() for r in CUE_RE for m in [r.search(text)] if m})
    results = []
    if urls:
        for u, p in zip(urls, MODEL.predict_proba(urls)):
            results.append({
                "url": u,
                "prob": float(p),
                "phishing": bool(p >= 0.5),
                "signals": MODEL.explain(u) if p >= 0.5 else [],
            })
    url_score = max((r["prob"] for r in results), default=0.0)
    cue_score = min(0.15 * len(cues), 0.7) if len(text.split()) > 3 else 0.0
    score = max(url_score, cue_score)
    note = "" if urls else "No links found - verdict is based on wording cues only (heuristic)."
    return jsonify(
        verdict="phishing" if score >= 0.5 else "safe",
        score=score,
        results=results,
        cues=cues,
        note=note,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
