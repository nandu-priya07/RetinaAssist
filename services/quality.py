"""Preliminary technical image checks.

These are simple heuristics (darkness, brightness, near-empty content, blur),
NOT a validated retinal image-quality model. They cannot reliably detect every
unusable image, and a "pass" does not guarantee the image is gradable.
"""
import cv2
import numpy as np

DEFAULT_THRESHOLDS = {
    "dark_threshold": 25.0,      # mean brightness of visible area below this -> fail
    "bright_threshold": 230.0,   # mean brightness of visible area above this -> fail
    "min_content_fraction": 0.15,  # fraction of non-black pixels below this -> fail
    "blur_threshold": 15.0,      # Laplacian variance below this -> warning only
}


def check_quality(img, thresholds=None):
    """Return {'status': 'pass'|'warning'|'fail', 'checks': [...], 'metrics': {...}}."""
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update({k: float(v) for k, v in thresholds.items() if k in t})

    rgb = np.asarray(img.convert("RGB"))
    # Work on a downsized copy so checks are fast and resolution-independent.
    scale = 512.0 / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Fundus photos have a black border; measure brightness inside the visible area only.
    visible = gray > 15
    content_fraction = float(visible.mean())
    mean_visible = float(gray[visible].mean()) if visible.any() else 0.0
    overall_std = float(gray.std())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    checks = []

    def add(name, status, message):
        checks.append({"name": name, "status": status, "message": message})

    if content_fraction < t["min_content_fraction"] or overall_std < 3:
        add("Visible content", "fail",
            f"Only {content_fraction:.0%} of the image has visible content; the image appears nearly empty.")
    else:
        add("Visible content", "pass", f"{content_fraction:.0%} of the image has visible content.")

    if mean_visible < t["dark_threshold"]:
        add("Brightness", "fail", f"Image is extremely dark (mean brightness {mean_visible:.0f}/255).")
    elif mean_visible > t["bright_threshold"]:
        add("Brightness", "fail", f"Image is extremely bright / overexposed (mean brightness {mean_visible:.0f}/255).")
    else:
        add("Brightness", "pass", f"Mean brightness {mean_visible:.0f}/255 is within the accepted range.")

    if blur_var < t["blur_threshold"]:
        add("Sharpness (experimental)", "warning",
            f"Low edge detail (Laplacian variance {blur_var:.1f}); image may be blurred.")
    else:
        add("Sharpness (experimental)", "pass", f"Edge detail score {blur_var:.1f}.")

    statuses = {c["status"] for c in checks}
    status = "fail" if "fail" in statuses else "warning" if "warning" in statuses else "pass"
    return {
        "status": status,
        "checks": checks,
        "metrics": {
            "content_fraction": round(content_fraction, 3),
            "mean_brightness": round(mean_visible, 1),
            "blur_variance": round(blur_var, 1),
        },
    }
