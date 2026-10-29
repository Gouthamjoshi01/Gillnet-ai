# 🎣 Gillnet-ai

> An AI-powered phishing detection engine with an interactive fishing-themed interface.

![Gillnet-ai banner](assets/preview.svg)

**Gillnet-ai** analyzes suspicious URLs and raw email bodies to detect phishing threats in real time. Paste a URL or message, cast your line, and let the detection engine hook, evaluate, and reel in malicious threats.

---

## 📸 Demo

![Gillnet-ai web interface](assets/screenshot.png)

---

## ✨ Features

- **Multi-Vector Analysis:** Inspects raw URLs, domain structures, and full email content.
- **Ensemble Machine Learning:** Combines lexical heuristics with n-gram linguistic features.
- **Interactive UI:** Dynamic, fishing-themed dashboard displaying threat metrics and classification confidence.
- **Dual Interface:** Accessible via a lightweight web app (`app.py`) or scriptable CLI (`gillnet.py`).

---

## 🧠 How It Works

Gillnet-ai relies on an ensemble classification pipeline:

1. **Structural Heuristics (~23 Features):** Extracts URL/domain characteristics including Shannon entropy, suspicious TLDs, IP-based hostnames, excessive subdomains, redirect triggers, and character anomalies.
2. **Linguistic Analysis:** Evaluates character and word n-grams using Logistic Regression to capture subtle typosquatting and deceptive lexical patterns.
3. **Probability Ensemble:** Blends predictions from a **Random Forest** classifier and the **n-gram Logistic Regression** model to produce a final weighted threat probability score.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### 1. Clone & Install

```bash
git clone [https://github.com/Gouthamjoshi01/Gillnet-ai.git](https://github.com/Gouthamjoshi01/Gillnet-ai.git)
cd Gillnet-ai
pip install -r requirements.txt
