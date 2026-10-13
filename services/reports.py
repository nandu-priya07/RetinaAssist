"""Clinical Report Generator: builds a PDF from a stored case using ReportLab."""
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.predictor import CLASS_NAMES

NAVY = colors.HexColor("#0B2545")
TEAL = colors.HexColor("#0F8B8D")
LIGHT = colors.HexColor("#EEF4F7")
MUTED = colors.HexColor("#5B6B7C")

DISCLAIMER = (
    "RESEARCH PROTOTYPE - NOT A CLINICALLY VALIDATED DIAGNOSTIC SYSTEM. This report was produced by an "
    "AI-assisted screening tool for research and demonstration purposes using synthetic patient codes. "
    "The predicted grade is a model output, not a diagnosis. Class scores are raw model outputs, not "
    "calibrated disease probabilities. Grad-CAM highlights regions that influenced the model and does not "
    "confirm the presence of any lesion. Priority flags are configurable operational rules, not a validated "
    "clinical risk score. All results must be reviewed by a qualified clinician. The model has not been "
    "externally validated and has no regulatory approval."
)


def build_report(case, upload_root, out_dir, review_labels, generated_by):
    """Write the PDF for `case` (a dict from database.case_dict) and return the filename."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{case['case_code']}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    path = os.path.join(out_dir, filename)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], textColor=NAVY, fontSize=20, leading=24, alignment=0, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=TEAL, fontSize=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=12, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10, textColor=MUTED)
    warn = ParagraphStyle("warn", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#7A4B00"),
                          backColor=colors.HexColor("#FFF6E0"), borderPadding=6)

    def table(data, widths, header=True):
        t = Table(data, colWidths=widths)
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DEE6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        else:
            style += [("BACKGROUND", (0, 0), (0, -1), LIGHT)]
        t.setStyle(TableStyle(style))
        return t

    story = [
        Paragraph("RetinaAssist", h1),
        Paragraph("AI-assisted retinal screening report &mdash; research prototype", sub),
        Spacer(1, 6),
        Paragraph(DISCLAIMER, warn),
        Spacer(1, 8),
        Paragraph("Case details", h2),
        table([
            ["Case identifier", case["case_code"]],
            ["Synthetic patient code", case["patient_code"]],
            ["Screening date", case["screened_at"]],
            ["Clinician review status", review_labels.get(case["review_status"], case["review_status"])],
            ["Report generated", f"{datetime.now():%Y-%m-%d %H:%M} by {generated_by}"],
        ], [55 * mm, 115 * mm], header=False),
    ]

    # Images: original and Grad-CAM overlay side by side.
    img_w = 82 * mm
    cells, captions = [], []
    for rel, caption in [(case["image_path"], "Original fundus image"),
                         (case.get("overlay_path"), "Grad-CAM overlay (model influence)")]:
        full = os.path.join(upload_root, rel) if rel else None
        if full and os.path.exists(full):
            from PIL import Image as PILImage
            with PILImage.open(full) as im:
                ratio = im.height / im.width
            cells.append(RLImage(full, width=img_w, height=img_w * ratio))
        else:
            cells.append(Paragraph("Not available" + (f" ({case['gradcam_error']})" if case.get("gradcam_error") else ""), small))
        captions.append(Paragraph(caption, small))
    story += [Paragraph("Images", h2), Table([cells, captions], colWidths=[85 * mm, 85 * mm])]

    story += [
        Paragraph("Model output", h2),
        Paragraph(f"<b>Predicted grade:</b> {case['predicted_grade']} &mdash; {case['predicted_label']} "
                  f"(model class score {case['top_score']:.1%}; not a calibrated probability)", body),
        Spacer(1, 4),
    ]
    score_rows = [["Grade", "Class", "Model class score"]]
    for i, (name, score) in enumerate(zip(CLASS_NAMES, case["scores"])):
        score_rows.append([str(i), name + ("  (predicted)" if i == case["predicted_grade"] else ""), f"{score:.1%}"])
    story.append(table(score_rows, [20 * mm, 90 * mm, 60 * mm]))

    q = case["quality"]
    story += [Paragraph(f"Preliminary image checks: {q['status'].upper()}", h2)]
    story.append(table([["Check", "Result", "Detail"]] +
                       [[c["name"], c["status"].upper(), Paragraph(c["message"], body)] for c in q["checks"]],
                       [45 * mm, 20 * mm, 105 * mm]))
    story.append(Paragraph("These are simple technical heuristics, not a validated retinal image-quality model.", small))

    story += [Paragraph(f"Review priority: {case['priority'].upper()} (operational, not medical severity)", h2)]
    if case["flags"]:
        story.append(table([["Flag", "Reason"]] + [[f["title"], Paragraph(f["reason"], body)] for f in case["flags"]],
                           [50 * mm, 120 * mm]))
    else:
        story.append(Paragraph("No review flags were raised by the configured rules.", body))

    if case.get("clinician_assessment") or case.get("notes"):
        story.append(Paragraph("Clinician review", h2))
        rows = []
        if case.get("clinician_assessment"):
            rows.append(["Assessment of AI output", case["clinician_assessment"].replace("_", " ").capitalize()])
        if case.get("clinician_grade") is not None:
            rows.append(["Clinician-recorded grade", f"{case['clinician_grade']} - {CLASS_NAMES[case['clinician_grade']]}"])
        if case.get("notes"):
            rows.append(["Clinical notes", Paragraph(case["notes"].replace("\n", "<br/>"), body)])
        story.append(table(rows, [55 * mm, 115 * mm], header=False))

    story += [Spacer(1, 12), Paragraph(DISCLAIMER, small)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 10 * mm, f"RetinaAssist research prototype | {case['case_code']} | Not for clinical use")
        canvas.drawRightString(190 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"RetinaAssist report {case['case_code']}", author="RetinaAssist prototype")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return filename
