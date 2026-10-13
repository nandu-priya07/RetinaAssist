# RetinaAssist — Explainable AI for Retinal Screening (research prototype)

RetinaAssist is a Flask web application that grades retinal fundus photographs for diabetic retinopathy with a
fine-tuned EfficientNetB0 model, explains each result with Grad-CAM, and supports clinician review with
transparent priority flags, case records, PDF reports and an audit trail.

[📂 GitHub Repository](https://github.com/nandu-priya07/RetinaAssist)

**Quick links:** [Overview](#overview) · [Features](#features) · [Technology stack](#technology-stack) ·
[Installation and local setup](#installation-and-local-setup) · [Usage](#usage) · [Deployment](#deployment) ·
[Responsible use and limitations](#responsible-use-and-limitations)

> **Responsible use:** RetinaAssist is a research prototype for diabetic retinopathy screening and has **not been
> clinically validated**. Its output must not replace professional medical judgement. Use synthetic patient codes only.

## Overview

Screening programmes produce large volumes of retinal images that specialists must review. RetinaAssist explores how
explainable AI could help teams organise that workload: each uploaded image receives a predicted grade with all five
class scores, a Grad-CAM visualisation, preliminary image checks and rule-based review flags, and a clinician makes the
final decision.

## Features

- **Five-grade classification** — No DR, Mild, Moderate, Severe and Proliferative DR, with all five class scores.
- **Explainable AI** — Grad-CAM heatmaps and overlays of the regions that influenced the model output.
- **Image-quality checks and review flags** — technical checks plus configurable, transparent priority and uncertainty rules.
- **Clinician-in-the-loop review** — ophthalmologists record their assessment, grade and notes.
- **Screening records and PDF reports** — searchable case history and downloadable reports.
- **Access control and accountability** — role-based permissions and an append-only audit trail.

## Technology stack

Python · Flask · TensorFlow/Keras (EfficientNetB0) · NumPy · Pillow · OpenCV · SQLite · ReportLab ·
Bootstrap 5 · HTML/CSS/JavaScript · Gunicorn and Docker for deployment.

## Installation and local setup

Requires Python 3.13 (the version used for development; TensorFlow 2.21 supports 3.10–3.13).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The model loads at startup (~10–20 s).
If activation is blocked on Windows: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Optional settings are listed in `.env.example` (copy to `.env`).

## Usage

1. Sign in with a demo account (below).
2. **New screening** — enter a synthetic patient code (e.g. `SYN-0001`) and upload a PNG/JPEG fundus image.
3. **AI analysis** — review the predicted grade, all five scores, Grad-CAM views, image checks and review flags.
4. As `doctor`, record a review and **generate a PDF report**; explore **Case records**, **Reports** and **Analytics**.
5. As `admin`, manage users, review rules and the audit trail.

### Demo accounts

| Username | Role | Can do |
|---|---|---|
| `staff` | Screening staff | New screenings, view cases |
| `doctor` | Ophthalmologist | Screenings, clinical review, reports, analytics |
| `admin` | Administrator | Users, review rules, audit trail, reports, analytics |

Locally, all three use `RETINAASSIST_DEMO_PASSWORD` (local default `RetinaDemo2026!`), applied when the database is
first created; `RETINAASSIST_ADMIN_PASSWORD` optionally gives `admin` its own password. Change passwords in **Settings**.
New sign-ups via **Request access** stay inactive until an admin approves them.

On a public deployment the default password is never used: the app refuses to start unless both passwords are set as
secrets, and the admin password must differ from the demo password. Demo sign-in details are shared by the project
owner on request; the admin password is never shared.

### Tests

```powershell
python -m unittest discover -s tests -v
```

Uses the real model and the image(s) in `test_images/`. The bundled `synthetic_fundus.png` is a generated, synthetic
test fixture (not a patient image), so its predictions are meaningless — it only exercises the pipeline.

## Deployment

The app is not currently deployed. It is prepared for any Docker host that can run TensorFlow on CPU
(plan for at least 2 GB of RAM; small free tiers with ~512 MB are not enough).

- The root `Dockerfile` builds an image from this repository, verifies the model's SHA-256 checksum, and starts
  `gunicorn wsgi:app` on the port given by the `PORT` environment variable (default 7860):

  ```bash
  docker build -t retinaassist .
  docker run -p 7860:7860 -e RETINAASSIST_SECRET_KEY=... -e RETINAASSIST_DEMO_PASSWORD=...              -e RETINAASSIST_ADMIN_PASSWORD=... retinaassist
  ```

- Required **secrets** (set them in your host's secret/environment settings; never commit them):
  - `RETINAASSIST_SECRET_KEY` — long random string used to sign session cookies.
  - `RETINAASSIST_DEMO_PASSWORD` — password for the `doctor` and `staff` demo accounts.
  - `RETINAASSIST_ADMIN_PASSWORD` — a different password for the `admin` account.
- `RETINAASSIST_ENV=production` is set in the image: the app refuses to start without the secrets above, sends the
  session cookie over HTTPS only, and runs without debug mode.

**Storage:** the SQLite database, uploaded images and PDF reports live in `instance/` inside the container. On hosts
with temporary disks they are reset on every restart or redeploy, and demo data is visible to everyone using the shared
demo accounts — never upload real patient images. Persistent storage needs a mounted volume or an external database.

## Responsible use and limitations

- Research prototype; **not clinically validated** and not approved for diagnostic use.
- Predictions are model outputs, not diagnoses; class scores are raw softmax outputs, not calibrated probabilities.
- Grad-CAM shows model influence, not confirmed lesions.
- Image checks are simple heuristics; priority and uncertainty thresholds are configurable and not clinically validated.
- Use synthetic patient codes and authorised test images only — never real patient data.

## Project structure

- `app.py` — Flask routes, authentication, role checks; `wsgi.py` — production entry point
- `services/predictor.py` — validation, preprocessing (224×224, raw 0–255; the model contains its own rescaling), inference
- `services/gradcam.py` — Grad-CAM on `efficientnetb0/top_conv`
- `services/quality.py` — preliminary image heuristics (not a validated quality model)
- `services/priority.py` — configurable review rules (operational priority, not severity)
- `services/pipeline.py` — end-to-end screening + storage
- `services/reports.py` — ReportLab PDF reports
- `database/db.py` — SQLite schema; audit log is append-only (DB triggers)
- `instance/` — database, uploads, reports, secret key (git-ignored, created at runtime)

## Illustrations

Images in `static/images/` are decorative **artwork**, labelled "Illustration" in the UI. They are never sent to the
model or shown as patient images, predictions or Grad-CAM output. Generated artwork comes from
`python tools/generate_illustrations.py`; the Home page eye is based on `tools/assets/hero-eye-source.png`.

## Reset the demo data

Stop the app and delete the `instance/` folder. It is recreated on the next start.

## Model

`models/retinaassist_efficientnet_finetuned.keras` — EfficientNetB0 fine-tuned on APTOS 2019
(SHA-256 `48bec19cb6a312e82f844f20258d9cf7a82ab26dc1b84a461e36777bd835810a`).
Offline validation (Colab): accuracy ≈ 79.8%, macro F1 ≈ 0.595. A class-weighted MobileNetV2 reached a higher
macro F1 (≈ 0.614). No external clinical validation; no regulatory approval. A real clinical deployment would require
external validation, privacy safeguards, a security review and regulatory assessment.
