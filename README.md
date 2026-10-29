# 🎣 Gillnet-ai

> Real-time phishing detection engine and threat triage platform combining dynamic network forensics with an ensemble machine learning pipeline.

![Gillnet-ai Banner](assets/preview.svg)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Engine: Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

**Gillnet-ai** analyzes suspicious URLs, domain structures, and raw message payloads to identify credential harvesters and deceptive web infrastructure in real time. It features an interactive fishing-themed web dashboard and an automated CLI designed for security operations and triage workflows.

---

## 📸 Interface Preview

![Gillnet-ai Web Interface](assets/screenshot.png)

---

## ✨ Features

- **Dynamic Network Telemetry:** Performs live A-record DNS resolution, SSL/TLS certificate verification, and tracks open HTTP redirect hops.
- **Domain Age Profiling:** Queries WHOIS records to identify newly registered domains (<30 days), a key indicator of ephemeral phishing kits.
- **Ensemble Machine Learning:** Blends a **Random Forest** classifier trained on 13+ structural lexical features with an **n-gram Logistic Regression** pipeline targeting typosquatting patterns.
- **Explainable Triage Signals:** Flags actionable detection indicators (e.g., raw IP hostnames, excessive subdomain depth, elevated Shannon entropy, deceptive credential keywords).
- **Dual Mode Deployment:** Accessible via an interactive web dashboard (Flask/Gunicorn) or a scriptable terminal CLI.

---

## 🧠 Detection Architecture

```text
Incoming Target URL
        │
        ├──► [Feature Extraction]
