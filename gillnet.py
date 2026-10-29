#!/usr/bin/env python3
"""
Gillnet-ai Core Engine: Network Forensics, Heuristic Scoring, ML Classification, and CLI.
"""
import argparse
import datetime
import math
import os
import re
import socket
import ssl
from urllib.parse import urlparse

import dns.resolver
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import whois

MODEL_PATH = "gillnet_model.joblib"

SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".club", ".work", ".info", ".buzz", ".fit",
    ".tk", ".ml", ".ga", ".cf", ".gq", ".country", ".kim", ".cricket", ".icu"
}

KEYWORDS = [
    "login", "verify", "account", "update", "secure", "banking",
    "signin", "confirm", "service", "password", "support", "billing",
    "wallet", "authenticate", "portal", "recovery", "security"
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
        self.rf = RandomForestClassifier(n_estimators=150, random_state=42)
        self.text_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=3000)),
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
        return ((0.6 * rf_probs) + (0.4 * text_probs)).tolist()


class ThreatAnalyzer:
    """Combines ML probabilities with dynamic network triage and WHOIS profiling."""

    def __init__(self, model: PhishModel):
        self.model = model

    def check_network_indicators(self, url: str) -> dict:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or ""
        indicators = {
            "resolved_ip": None,
            "has_dns": False,
            "domain_age_days": None,
            "ssl_valid": False,
            "redirect_count": 0,
            "signals": []
        }

        if not host:
            return indicators

        # 1. DNS Resolution
        try:
            answers = dns.resolver.resolve(host, "A")
            indicators["resolved_ip"] = answers[0].to_text()
            indicators["has_dns"] = True
        except Exception:
            indicators["signals"].append("Failed DNS resolution (Unmapped or dead host)")

        # 2. WHOIS Domain Age Analysis
        try:
            w = whois.whois(host)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date:
                age_days = (datetime.datetime.now() - creation_date).days
                indicators["domain_age_days"] = age_days
                if age_days < 30:
                    indicators["signals"].append(f"Newly registered domain ({age_days} days old)")
        except Exception:
            pass

        # 3. SSL / Certificate Verification
        if parsed.scheme == "https":
            try:
                ctx = ssl.create_default_context()
                with socket.create_connection((host, 443), timeout=3) as sock:
                    with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                        cert = ssock.getpeercert()
                        indicators["ssl_valid"] = bool(cert)
            except Exception:
                indicators["signals"].append("Invalid or untrusted TLS/SSL certificate")

        # 4. Redirect Chain Tracing
        try:
            res = requests.head(url if "://" in url else f"http://{url}", timeout=3, allow_redirects=True)
            indicators["redirect_count"] = len(res.history)
            if len(res.history) > 2:
                indicators["signals"].append(f"Abnormal redirect chain ({len(res.history)} hops)")
        except Exception:
            pass

        return indicators

    def analyze(self, url: str) -> dict:
        ml_prob = self.model.predict_proba([url])[0]
        net_info = self.check_network_indicators(url)
        signals = list(net_info["signals"])

        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or ""

        # Heuristic rules
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", host):
            signals.append("Host is a raw IP address")
        if any(host.endswith(t) for t in SUSPICIOUS_TLDS):
            signals.append("High-risk or uncommon TLD")
        if sum(1 for kw in KEYWORDS if kw in url.lower()) >= 2:
            signals.append("Multiple targeted authentication keywords")
        if shannon_entropy(host) > 3.8:
            signals.append("High hostname entropy (possible DGA/typosquatting)")

        # Composite Scoring: Blend ML with dynamic network telemetry
        score = ml_prob
        if net_info["domain_age_days"] is not None and net_info["domain_age_days"] < 30:
            score = max(score, 0.85)
        if not net_info["has_dns"]:
            score = max(score, 0.65)

        verdict = "PHISHING" if score >= 0.5 else "BENIGN"
        return {
            "target": url,
            "verdict": verdict,
            "risk_score": round(float(score), 4),
            "ml_probability": round(float(ml_prob), 4),
            "signals": signals,
            "network_telemetry": {
                "ip": net_info["resolved_ip"],
                "domain_age_days": net_info["domain_age_days"],
                "redirects": net_info["redirect_count"],
                "ssl_valid": net_info["ssl_valid"]
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Gillnet-ai CLI Phishing & Threat Triage Engine")
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--data", default="sample_data.csv", help="Path to training CSV dataset")

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("url", help="Target URL to inspect")

    args = parser.parse_args()

    if args.command == "train":
        if not os.path.exists(args.data):
            print(f"[-] Missing dataset: {args.data}")
            return
        df = pd.read_csv(args.data)
        model = PhishModel().fit(df["url"].astype(str).tolist(), df["label"].astype(int).values)
        joblib.dump(model, MODEL_PATH)
        print(f"[+] Serialized model generated: {MODEL_PATH}")

    elif args.command == "predict":
        if not os.path.exists(MODEL_PATH):
            print("[-] No model found. Please train first.")
            return
        model = joblib.load(MODEL_PATH)
        analyzer = ThreatAnalyzer(model)
        result = analyzer.analyze(args.url)

        print("=" * 60)
        print(f"TARGET      : {result['target']}")
        print(f"VERDICT     : {result['verdict']} (Score: {result['risk_score'] * 100:.1f}%)")
        print(f"RESOLVED IP : {result['network_telemetry']['ip'] or 'N/A'}")
        print(f"DOMAIN AGE  : {result['network_telemetry']['domain_age_days'] or 'Unknown'} days")
        print(f"SIGNALS     : {', '.join(result['signals']) if result['signals'] else 'None'}")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
