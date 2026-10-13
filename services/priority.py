"""Clinical Priority & Uncertainty Engine (transparent, rule-based).

Priority here is OPERATIONAL (how soon a clinician should look at the case),
not a measure of medical severity and not a validated clinical risk score.
All thresholds are clinic-configurable and have NOT been clinically validated.
"""
from services.predictor import CLASS_NAMES

PRIORITY_ORDER = {"high": 0, "medium": 1, "routine": 2}


def _parse_grades(text):
    grades = set()
    for part in str(text).split(","):
        part = part.strip()
        if part.isdigit() and 0 <= int(part) < len(CLASS_NAMES):
            grades.add(int(part))
    return grades


def evaluate(prediction, quality, settings):
    """Return (priority, flags). Each flag has a code, title and the reason it fired."""
    flags = []
    margin_limit = float(settings["uncertainty_margin"])
    min_top = float(settings["min_top_score"])
    priority_grades = _parse_grades(settings["priority_grades"])

    if quality["status"] == "fail":
        failed = [c["message"] for c in quality["checks"] if c["status"] == "fail"]
        flags.append({
            "code": "recapture",
            "title": "Image needs recapture",
            "reason": "Preliminary image checks failed: " + " ".join(failed),
        })
    elif quality["status"] == "warning":
        warned = [c["message"] for c in quality["checks"] if c["status"] == "warning"]
        flags.append({
            "code": "quality_warning",
            "title": "Image quality warning",
            "reason": " ".join(warned) + " Consider recapturing if the image is not gradable.",
        })

    uncertain_reasons = []
    if prediction["margin"] < margin_limit:
        uncertain_reasons.append(
            f"top two class scores ({CLASS_NAMES[prediction['grade']]} {prediction['top_score']:.1%} vs "
            f"{CLASS_NAMES[prediction['second_grade']]} {prediction['second_score']:.1%}) differ by "
            f"{prediction['margin']:.1%}, below the configured margin of {margin_limit:.0%}"
        )
    if prediction["top_score"] < min_top:
        uncertain_reasons.append(
            f"highest class score {prediction['top_score']:.1%} is below the configured minimum of {min_top:.0%}"
        )
    if uncertain_reasons:
        flags.append({
            "code": "uncertain",
            "title": "Specialist review recommended",
            "reason": "Experimental uncertainty rule: " + "; ".join(uncertain_reasons) + ".",
        })

    if prediction["grade"] in priority_grades:
        configured = ", ".join(CLASS_NAMES[g] for g in sorted(priority_grades))
        flags.append({
            "code": "priority_grade",
            "title": "Higher review priority",
            "reason": f"Model output '{prediction['label']}' matches the clinic review rule "
                      f"for prioritised grades ({configured}).",
        })

    codes = {f["code"] for f in flags}
    if codes & {"priority_grade", "recapture"}:
        priority = "high"
    elif codes & {"uncertain", "quality_warning"}:
        priority = "medium"
    else:
        priority = "routine"
    return priority, flags
