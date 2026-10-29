# Gillnet-ai

AI phishing detector with a fishing-themed web UI. Paste a URL or a whole email, Gillnet-ai casts a line, and if it's phishing, you'll see the phish get reeled in.

![Gillnet-ai Banner](assets/preview.png)

## Overview & Demo

<!-- Tip: Drag & drop your screenshot directly into GitHub's editor or replace the link below with your image path/URL -->
![Gillnet-ai Web Interface Screenshot](assets/screenshot.png)

## Run

```bash
pip install -r requirements.txt
python gillnet.py train --data your_dataset.csv   # columns: url,label (1=phish, 0=legit)
python app.py                                    # [http://127.0.0.1:5000](http://127.0.0.1:5000)
