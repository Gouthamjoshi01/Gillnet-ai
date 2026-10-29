#!/usr/bin/env python3
"""
Gillnet-ai - AI Phishing URL Detector
------------------------
Train:    python gillnet.py train --data data.csv
Predict:  python gillnet.py predict "http://paypa1-secure-login.xyz/verify"
Batch:    python gillnet.py predict --file urls.txt

data.csv needs two columns: url,label   (label: 1 = phishing, 0 = legitimate)
"""
import argparse
import math
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = "gillnet_model.joblib"

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank",
    "confirm", "password", "wallet", "free", "bonus", "support", "billing",
    "paypal", "apple", "microsoft", "amazon", "office365", "invoice",
]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
RISKY_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "zip", "work", "support", "country"}
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum((c / len(s)) * math.log2(c / len(s)) for c in counts.values())


def normalize(url: str) -> str:
    url = str(url).strip()
    return url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url


def extract_features(url: str) -> dict:
    url = normalize(url)
    p = urlparse(url)
    host = (p.hostname or "").lower()
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    low = url.lower()
    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(p.path),
        "query_len": len(p.query),
        "n_dots": host.count("."),
        "n_hyphens_host": host.count("-"),
        "n_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "n_subdomains": max(len(parts) - 2, 0),
        "has_ip": int(bool(IP_RE.match(host))),
        "has_at": int("@" in url),
        "has_https": int(p.scheme == "https"),
        "has_port": int(p.port is not None),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(host in SHORTENERS),
        "risky_tld": int(tld in RISKY_TLDS),
        "n_susp_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "n_special": sum(url.count(c) for c in "?&=%_~#"),
        "n_slashes": p.path.count("/"),
        "double_slash_path": int("//" in p.path),
        "host_entropy": entropy(host),
        "url_entropy": entropy(url),
        "brand_in_subdomain": int(any(b in ".".join(parts[:-2]) for b in ("paypal", "apple", "amazon", "microsoft", "google", "bank"))),
    }


class PhishModel:
    """Random Forest on hand-crafted features + Logistic Regression on char n-grams,
    blended by averaging probabilities."""

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced")
        self.tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100_000)
        self.lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.cols = None

    def _feats(self, urls):
        df = pd.DataFrame([extract_features(u) for u in urls])
        if self.cols is None:
            self.cols = list(df.columns)
        return df[self.cols]

    def fit(self, urls, y):
        self.rf.fit(self._feats(urls), y)
        self.lr.fit(self.tfidf.fit_transform([normalize(u) for u in urls]), y)
        return self

    def predict_proba(self, urls):
        p_rf = self.rf.predict_proba(self._feats(urls))[:, 1]
        p_lr = self.lr.predict_proba(self.tfidf.transform([normalize(u) for u in urls]))[:, 1]
        return (p_rf + p_lr) / 2

    def explain(self, url, top=5):
        """Rough explanation: features with highest RF importance that are 'active' for this URL."""
        f = extract_features(url)
        imp = dict(zip(self.cols, self.rf.feature_importances_))
        flagged = []
        for k, v in f.items():
            if v and k in ("has_ip", "has_at", "has_punycode", "is_shortener", "risky_tld",
                           "brand_in_subdomain", "double_slash_path", "n_susp_words", "n_hyphens_host", "n_subdomains"):
                flagged.append((imp[k], f"{k}={v}"))
        return [t for _, t in sorted(flagged, reverse=True)[:top]]


def cmd_train(args):
    df = pd.read_csv(args.data).dropna(subset=["url", "label"]).drop_duplicates("url")
    X, y = df["url"].astype(str).tolist(), df["label"].astype(int).values
    print(f"Loaded {len(df)} URLs ({y.sum()} phishing / {len(y) - y.sum()} legit)")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = PhishModel().fit(Xtr, ytr)
    pred = (model.predict_proba(Xte) >= 0.5).astype(int)
    print(classification_report(yte, pred, target_names=["legit", "phishing"], digits=3))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(yte, pred))
    joblib.dump(model, args.model)
    print(f"Saved model -> {args.model}")


def cmd_predict(args):
    model = joblib.load(args.model)
    urls = list(args.urls)
    if args.file:
        urls += [l.strip() for l in open(args.file) if l.strip()]
    if not urls:
        sys.exit("No URLs given.")
    for u, p in zip(urls, model.predict_proba(urls)):
        verdict = "PHISHING" if p >= 0.5 else "legit"
        why = ", ".join(model.explain(u)) if p >= 0.5 else ""
        print(f"[{verdict:8}] {p:6.1%}  {u}" + (f"\n           signals: {why}" if why else ""))


def main():
    ap = argparse.ArgumentParser(description="Gillnet-ai - AI phishing URL detector")
    ap.add_argument("--model", default=MODEL_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("urls", nargs="*")
    p.add_argument("--file")
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    # import as a module so pickled models reference gillnet.PhishModel (loadable from app.py)
    import gillnet
    gillnet.main()
