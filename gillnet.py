#!/usr/bin/env python3
"""Gillnet-ai ML engine: feature extraction, ensemble model, training, and CLI."""
import argparse
import math
import os
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = "gillnet_model.joblib"

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".info", ".buzz", ".fit",
    ".tk", ".ml", ".ga", ".cf", ".gq", ".country", ".kim", ".cricket"
}

KEYWORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "signin", "confirm", "service", "password", "support", "billing"
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    ent = 0.0
    for count in freq.values():
        p = count / len(s)
        ent -= p * math.log2(p)
    return ent


def extract_features(url: str) -> list:
    url_lower = url.lower()
    if not re.match(r"^https?://", url_lower):
        parsed = urlparse("http://" + url)
    else:
        parsed = urlparse(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    url_len = len(url)
    host_len = len(hostname)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    at_count = url.count("@")
    num_digits = sum(c.isdigit() for c in url)
    digit_ratio = num_digits / max(url_len, 1)

    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    is_ip = 1 if re.match(ip_pattern, hostname) else 0

    suspicious_tld = 1 if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0
    subdomains = max(len(hostname.split(".")) - 2, 0)
    has_redirect = 1 if "//" in path else 0
    entropy = shannon_entropy(hostname)
    keyword_hits = sum(1 for kw in KEYWORDS if kw in url_lower)
    is_https = 1 if parsed.scheme == "https" else 0

    return [
        url_len, host_len, dot_count, hyphen_count, at_count,
        digit_ratio, is_ip, suspicious_tld, subdomains,
        has_redirect, entropy, keyword_hits, is_https
    ]


class PhishModel:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.text_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=2000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])

    def fit(self, urls: list, labels: np.ndarray):
        features = np.array([extract_features(u) for u in urls])
        self.rf.fit(features, labels)
        self.text_pipe.fit(urls, labels)
        return self

    def predict_proba(self, urls: list) -> list:
        features = np.array([extract_features(u) for u in urls])
        rf_probs = self.rf.predict_proba(features)[:, 1]
        text_probs = self.text_pipe.predict_proba(urls)[:, 1]
        ensemble_probs = (0.6 * rf_probs) + (0.4 * text_probs)
        return ensemble_probs.tolist()

    def explain(self, url: str) -> list:
        signals = []
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or ""

        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host):
            signals.append("Host is a raw IP address")
        if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
            signals.append("High-risk or uncommon TLD")
        if sum(1 for kw in KEYWORDS if kw in url.lower()) >= 2:
            signals.append("Multiple sensitive authentication keywords")
        if shannon_entropy(host) > 3.8:
            signals.append("High hostname entropy (possible DGA or typosquatting)")
        if len(host.split(".")) > 3:
            signals.append("Excessive subdomain depth")
        if "@" in url:
            signals.append("URL contains embedded credentials (@)")
        if not signals:
            signals.append("Anomalous structural patterns detected by ensemble classifier")
        return signals


def main():
    parser = argparse.ArgumentParser(description="Gillnet Phishing Classifier CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--data", default="sample_data.csv", help="Path to training dataset CSV")

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("url", help="Target URL to analyze")

    args = parser.parse_args()

    if args.command == "train":
        if not os.path.exists(args.data):
            print(f"[-] Data file {args.data} not found.")
            return
        df = pd.read_csv(args.data)
        model = PhishModel().fit(df["url"].astype(str).tolist(), df["label"].astype(int).values)
        joblib.dump(model, MODEL_PATH)
        print(f"[+] Model successfully trained and saved to {MODEL_PATH}")

    elif args.command == "predict":
        if not os.path.exists(MODEL_PATH):
            print("[-] Model file not found. Run train first.")
            return
        model = joblib.load(MODEL_PATH)
        prob = model.predict_proba([args.url])[0]
        status = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
        print(f"\nTarget: {args.url}")
        print(f"Status: {status} ({prob * 100:.2f}% risk)")
        if prob >= 0.5:
            print("Signals:", ", ".join(model.explain(args.url)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
