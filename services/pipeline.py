"""End-to-end screening: validate -> quality checks -> predict -> Grad-CAM -> rules -> store."""
import json
import os
import secrets
from datetime import datetime

from database.db import get_db, get_settings, log_event, now_iso
from services import gradcam, predictor, priority, quality


def new_case_code():
    return f"RA-{datetime.now():%Y%m%d}-{secrets.token_hex(3).upper()}"


def run_screening(file_bytes, filename, patient_code, upload_root, user_id, username=None):
    """Analyse one image with the real model and store the case. Returns the case code.

    Raises predictor.ImageValidationError for unusable uploads.
    """
    img = predictor.load_image(file_bytes, filename)
    settings = get_settings()

    q = quality.check_quality(img, settings)
    pred = predictor.predict(img)
    prio, flags = priority.evaluate(pred, q, settings)

    case_code = new_case_code()
    case_dir = os.path.join(upload_root, case_code)
    os.makedirs(case_dir, exist_ok=True)

    # Re-encode instead of saving the raw upload, so only clean pixel data is stored.
    heatmap_rel = overlay_rel = gradcam_error = None
    try:
        original, heat, overlay = gradcam.generate(img, pred["grade"])
        heat.save(os.path.join(case_dir, "heatmap.png"))
        overlay.save(os.path.join(case_dir, "overlay.png"))
        heatmap_rel, overlay_rel = f"{case_code}/heatmap.png", f"{case_code}/overlay.png"
    except Exception as exc:  # Grad-CAM failure must not lose the prediction
        original = img.copy()
        original.thumbnail((800, 800))
        gradcam_error = f"Grad-CAM could not be generated: {exc.__class__.__name__}"
    original.save(os.path.join(case_dir, "original.png"))

    ts = now_iso()
    db = get_db()
    db.execute(
        """INSERT INTO cases (case_code, patient_code, screened_at, image_path, heatmap_path, overlay_path,
               gradcam_error, predicted_grade, predicted_label, scores_json, top_score, score_margin,
               quality_status, quality_json, flags_json, priority, review_status, created_by,
               created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)""",
        (case_code, patient_code, ts, f"{case_code}/original.png", heatmap_rel, overlay_rel, gradcam_error,
         pred["grade"], pred["label"], json.dumps(pred["scores"]), pred["top_score"], pred["margin"],
         q["status"], json.dumps(q), json.dumps(flags), prio, user_id, ts, ts),
    )
    db.commit()

    kw = {"user_id": user_id, "username": username} if username else {}
    log_event("case_created", case_code, {"patient_code": patient_code}, **kw)
    log_event("analysis_succeeded", case_code,
              {"grade": pred["grade"], "top_score": round(pred["top_score"], 4),
               "quality": q["status"], "priority": prio, "gradcam": gradcam_error is None}, **kw)
    return case_code
