"""RetinaAssist - Explainable AI for Retinal Screening (research prototype).

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import re
import secrets
import sqlite3
from datetime import date, timedelta
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, abort, flash, g, redirect, render_template, request, send_from_directory,
                   session, url_for)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.security import check_password_hash, generate_password_hash

from database.db import (DEFAULT_SETTINGS, case_dict, get_db, get_settings, init_db, log_event, now_iso)
from services.predictor import CLASS_NAMES, MAX_FILE_BYTES, ImageValidationError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

ROLE_LABELS = {"doctor": "Ophthalmologist", "staff": "Screening staff", "admin": "Administrator"}
REVIEW_STATUSES = {
    "pending": "Pending review",
    "in_review": "Under review",
    "reviewed": "Reviewed",
    "referred": "Referred for specialist examination",
    "recapture": "Recapture requested",
}
ASSESSMENTS = {"agree": "Agrees with AI output", "disagree": "Disagrees with AI output",
               "undetermined": "Unable to determine"}
PRIORITY_LABELS = {"high": "High", "medium": "Medium", "routine": "Routine"}
QUALITY_LABELS = {"pass": "Passed", "warning": "Warning", "fail": "Failed"}

# Offline validation results reported from the Google Colab evaluation (APTOS 2019 validation split).
# These are NOT measured by this application and are not live production monitoring.
MODEL_METRICS = {
    "selected": {"name": "EfficientNetB0 (fine-tuned)", "file": "retinaassist_efficientnet_finetuned.keras",
                 "accuracy": 0.7981, "macro_f1": 0.595},
    "comparison": [
        {"name": "EfficientNetB0 (fine-tuned)", "accuracy": 0.7981, "macro_f1": 0.595, "in_use": True},
        {"name": "MobileNetV2 (class-weighted)", "accuracy": None, "macro_f1": 0.614, "in_use": False},
    ],
    "dataset": "APTOS 2019 Blindness Detection (3,662 labelled images, 5 grades)",
}

PATIENT_CODE_RE = re.compile(r"^[A-Za-z0-9-]{3,20}$")


def create_app(test_config=None):
    # Render mounts persistent storage at RETINAASSIST_DATA_DIR; local runs keep using instance/.
    data_dir = os.environ.get("RETINAASSIST_DATA_DIR") or os.path.join(BASE_DIR, "instance")
    app = Flask(__name__, instance_path=data_dir)
    os.makedirs(app.instance_path, exist_ok=True)

    app.config.update(
        DATABASE=os.path.join(app.instance_path, "retinaassist.db"),
        UPLOAD_ROOT=os.path.join(app.instance_path, "uploads"),
        REPORT_ROOT=os.path.join(app.instance_path, "reports"),
        MAX_CONTENT_LENGTH=MAX_FILE_BYTES + 512 * 1024,
        DEMO_PASSWORD=os.environ.get("RETINAASSIST_DEMO_PASSWORD", "RetinaDemo2026!"),
        # The admin account can have its own password so that sharing the demo password never grants admin access.
        ADMIN_PASSWORD=os.environ.get("RETINAASSIST_ADMIN_PASSWORD"),  # falls back to DEMO_PASSWORD when unset
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        CSRF_ENABLED=True,
        PRODUCTION=False,
    )
    if os.environ.get("RETINAASSIST_ENV") == "production":
        # Never run a public deployment with a missing secret or a publicly known password.
        missing = [k for k in ("RETINAASSIST_SECRET_KEY", "RETINAASSIST_DEMO_PASSWORD", "RETINAASSIST_ADMIN_PASSWORD")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError("Production requires these secrets to be set: " + ", ".join(missing))
        if os.environ["RETINAASSIST_ADMIN_PASSWORD"] == os.environ["RETINAASSIST_DEMO_PASSWORD"]:
            raise RuntimeError("RETINAASSIST_ADMIN_PASSWORD must differ from RETINAASSIST_DEMO_PASSWORD.")
        # Served over HTTPS: only send the session cookie over secure connections.
        app.config.update(SESSION_COOKIE_SECURE=True, PRODUCTION=True)
    if test_config:
        app.config.update(test_config)

    app.secret_key = os.environ.get("RETINAASSIST_SECRET_KEY") or app.config.get("SECRET_KEY") or _local_secret(app)
    os.makedirs(app.config["UPLOAD_ROOT"], exist_ok=True)
    os.makedirs(app.config["REPORT_ROOT"], exist_ok=True)
    init_db(app)
    register_routes(app)
    return app


def _local_secret(app):
    """Development fallback: a random key persisted in instance/ (never committed)."""
    path = os.path.join(app.instance_path, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


# ---------------------------------------------------------------- auth helpers

def current_user():
    if "user" not in g:
        g.user = None
        uid = session.get("user_id")
        if uid:
            row = get_db().execute("SELECT * FROM users WHERE id = ? AND is_active = 1", (uid,)).fetchone()
            if row is None:
                session.clear()
            g.user = row
    return g.user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please sign in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def roles_required(*roles):
    """Server-side role check. The role always comes from the database, never the browser."""
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user()["role"] not in roles:
                log_event("access_denied", details={"path": request.path})
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorator


def get_case_or_404(case_code):
    row = get_db().execute("SELECT * FROM cases WHERE case_code = ?", (case_code,)).fetchone()
    if row is None:
        abort(404)
    return case_dict(row)


# ---------------------------------------------------------------- routes

def register_routes(app):

    @app.before_request
    def csrf_protect():
        if request.method == "POST" and app.config["CSRF_ENABLED"]:
            token = session.get("csrf_token")
            if not token or token != request.form.get("csrf_token"):
                abort(400, description="Form expired or invalid. Please reload the page and try again.")

    @app.context_processor
    def inject_globals():
        def csrf_token():
            if "csrf_token" not in session:
                session["csrf_token"] = secrets.token_hex(16)
            return session["csrf_token"]
        return dict(current_user=current_user(), csrf_token=csrf_token, CLASS_NAMES=CLASS_NAMES,
                    ROLE_LABELS=ROLE_LABELS, REVIEW_STATUSES=REVIEW_STATUSES, ASSESSMENTS=ASSESSMENTS,
                    PRIORITY_LABELS=PRIORITY_LABELS, QUALITY_LABELS=QUALITY_LABELS)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_e):
        flash("The file is larger than 10 MB. Please upload a smaller image.", "danger")
        return redirect(url_for("screening"))

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def http_error(e):
        messages = {400: "Bad request", 403: "Access denied", 404: "Page not found"}
        return render_template("error.html", code=e.code, title=messages.get(e.code, "Error"),
                               description=e.description), e.code

    # ---------- public pages

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html", metrics=MODEL_METRICS)

    @app.route("/contact", methods=["GET", "POST"])
    def contact():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip()
            message = request.form.get("message", "").strip()
            if not name or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) or len(message) < 10:
                flash("Please provide your name, a valid email and a message of at least 10 characters.", "danger")
                return render_template("contact.html", form=request.form), 400
            db = get_db()
            db.execute("INSERT INTO contact_messages (name, email, organisation, message, created_at) VALUES (?, ?, ?, ?, ?)",
                       (name, email, request.form.get("organisation", "").strip(), message[:4000], now_iso()))
            db.commit()
            flash("Thank you. Your message was saved in the local demo database (no email is sent).", "success")
            return redirect(url_for("contact"))
        return render_template("contact.html", form={})

    # ---------- authentication

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user():
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            username = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "")
            row = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if row and check_password_hash(row["password_hash"], password):
                if not row["is_active"]:
                    flash("This account is awaiting administrator approval or has been deactivated.", "warning")
                    return render_template("login.html"), 403
                session.clear()
                session.permanent = True
                session["user_id"] = row["id"]
                session["username"] = row["username"]
                log_event("login", details={"role": row["role"]})
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
            log_event("login_failed", details={"username": username[:50]}, username="(anonymous)")
            flash("Invalid username or password.", "danger")
            return render_template("login.html"), 401
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        if current_user():
            log_event("logout")
        session.clear()
        flash("You have been signed out.", "info")
        return redirect(url_for("index"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        """Controlled demo registration: new accounts are inactive screening-staff accounts
        until an administrator approves them and assigns a role."""
        if request.method == "POST":
            username = request.form.get("username", "").strip().lower()
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            error = _validate_new_user(username, full_name, password, request.form.get("confirm", ""))
            if error:
                flash(error, "danger")
                return render_template("register.html", form=request.form), 400
            try:
                db = get_db()
                db.execute("INSERT INTO users (username, full_name, email, password_hash, role, is_active, created_at)"
                           " VALUES (?, ?, ?, ?, 'staff', 0, ?)",
                           (username, full_name, email, generate_password_hash(password), now_iso()))
                db.commit()
            except sqlite3.IntegrityError:
                flash("That username is already taken.", "danger")
                return render_template("register.html", form=request.form), 400
            log_event("account_requested", details={"username": username}, user_id=None, username=username)
            flash("Account request submitted. An administrator must approve it before you can sign in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html", form={})

    # ---------- app pages

    @app.route("/dashboard")
    @login_required
    def dashboard():
        db = get_db()
        stats = db.execute("""SELECT COUNT(*) total,
                   SUM(review_status IN ('pending','in_review')) awaiting,
                   SUM(priority = 'high' AND review_status IN ('pending','in_review')) high_open,
                   SUM(quality_status = 'fail') quality_fail
                   FROM cases""").fetchone()
        queue = [case_dict(r) for r in db.execute(
            """SELECT * FROM cases WHERE review_status IN ('pending','in_review')
               ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, screened_at
               LIMIT 6""")]
        recent = [case_dict(r) for r in db.execute("SELECT * FROM cases ORDER BY screened_at DESC, id DESC LIMIT 6")]
        today = db.execute("SELECT COUNT(*) FROM cases WHERE substr(screened_at, 1, 10) = ?",
                           (date.today().isoformat(),)).fetchone()[0]
        # Read-only status shown on the dashboard (no model performance figures here).
        from services import predictor
        system = {
            "model_file": os.path.exists(predictor.MODEL_PATH),
            "model_loaded": predictor._model is not None,
            "model_name": MODEL_METRICS["selected"]["name"],
            "settings": get_settings(),
        }
        return render_template("dashboard.html", stats=stats, queue=queue, recent=recent, today=today,
                               system=system, now=date.today())

    @app.route("/screening", methods=["GET", "POST"])
    @roles_required("staff", "doctor")
    def screening():
        if request.method == "POST":
            patient_code = request.form.get("patient_code", "").strip().upper()
            file = request.files.get("image")
            if not PATIENT_CODE_RE.match(patient_code):
                flash("Synthetic patient code must be 3-20 letters, digits or hyphens (e.g. SYN-0001).", "danger")
                return render_template("screening.html", patient_code=patient_code), 400
            if file is None or not file.filename:
                flash("Please choose a retinal fundus image to upload.", "danger")
                return render_template("screening.html", patient_code=patient_code), 400
            from services.pipeline import run_screening
            try:
                case_code = run_screening(file.read(), file.filename, patient_code, app.config["UPLOAD_ROOT"],
                                          current_user()["id"])
            except ImageValidationError as exc:
                log_event("analysis_failed", details={"reason": str(exc), "patient_code": patient_code})
                flash(str(exc), "danger")
                return render_template("screening.html", patient_code=patient_code), 400
            except Exception as exc:
                app.logger.exception("Analysis failed")
                log_event("analysis_failed", details={"reason": exc.__class__.__name__, "patient_code": patient_code})
                flash("The analysis could not be completed due to an internal error. Please try again.", "danger")
                return render_template("screening.html", patient_code=patient_code), 500
            flash("Analysis complete. The result requires clinician review.", "success")
            return redirect(url_for("analysis", case_code=case_code))
        return render_template("screening.html", patient_code="")

    @app.route("/cases/<case_code>")
    @login_required
    def analysis(case_code):
        case = get_case_or_404(case_code)
        db = get_db()
        log_event("case_viewed", case_code)
        reports = db.execute("""SELECT r.*, u.full_name FROM reports r LEFT JOIN users u ON u.id = r.generated_by
                                WHERE r.case_id = ? ORDER BY r.generated_at DESC""", (case["id"],)).fetchall()
        history = db.execute("SELECT * FROM audit_log WHERE case_code = ? AND event_type != 'case_viewed' ORDER BY id DESC",
                             (case_code,)).fetchall()
        reviewer = None
        if case["reviewed_by"]:
            reviewer = db.execute("SELECT full_name FROM users WHERE id = ?", (case["reviewed_by"],)).fetchone()
        return render_template("analysis.html", case=case, reports=reports, history=history, reviewer=reviewer)

    @app.route("/cases/<case_code>/review", methods=["POST"])
    @roles_required("doctor")
    def review_case(case_code):
        case = get_case_or_404(case_code)
        status = request.form.get("review_status", "")
        assessment = request.form.get("clinician_assessment") or None
        grade_raw = request.form.get("clinician_grade", "")
        notes = request.form.get("notes", "").strip()[:4000]
        if status not in REVIEW_STATUSES or (assessment and assessment not in ASSESSMENTS):
            abort(400)
        clinician_grade = int(grade_raw) if grade_raw.isdigit() and int(grade_raw) < len(CLASS_NAMES) else None

        db = get_db()
        db.execute("""UPDATE cases SET review_status = ?, clinician_assessment = ?, clinician_grade = ?, notes = ?,
                      reviewed_by = ?, updated_at = ? WHERE id = ?""",
                   (status, assessment, clinician_grade, notes, current_user()["id"], now_iso(), case["id"]))
        db.commit()
        if status != case["review_status"]:
            log_event("review_status_changed", case_code, {"from": case["review_status"], "to": status})
        if assessment and assessment != case["clinician_assessment"]:
            log_event(f"clinician_{assessment}", case_code,
                      {"ai_grade": case["predicted_grade"], "clinician_grade": clinician_grade})
        if notes != (case["notes"] or ""):
            log_event("notes_updated", case_code)
        flash("Review saved.", "success")
        return redirect(url_for("analysis", case_code=case_code))

    @app.route("/media/<case_code>/<kind>")
    @login_required
    def media(case_code, kind):
        if kind not in ("original.png", "heatmap.png", "overlay.png"):
            abort(404)
        get_case_or_404(case_code)
        return send_from_directory(os.path.join(app.config["UPLOAD_ROOT"], case_code), kind)

    @app.route("/records")
    @login_required
    def records():
        q = request.args.get("q", "").strip()
        grade = request.args.get("grade", "")
        status = request.args.get("status", "")
        prio = request.args.get("priority", "")
        quality = request.args.get("quality", "")
        sql, params = "SELECT * FROM cases WHERE 1=1", []
        if q:
            sql += " AND (case_code LIKE ? OR patient_code LIKE ?)"
            params += [f"%{q}%", f"%{q}%"]
        if grade.isdigit():
            sql += " AND predicted_grade = ?"
            params.append(int(grade))
        if status in REVIEW_STATUSES:
            sql += " AND review_status = ?"
            params.append(status)
        if prio in PRIORITY_LABELS:
            sql += " AND priority = ?"
            params.append(prio)
        if quality in QUALITY_LABELS:
            sql += " AND quality_status = ?"
            params.append(quality)
        sql += " ORDER BY screened_at DESC, id DESC LIMIT 500"
        cases = [case_dict(r) for r in get_db().execute(sql, params)]
        return render_template("records.html", cases=cases, filters=request.args)

    # ---------- reports

    @app.route("/reports")
    @roles_required("doctor", "admin")
    def reports():
        db = get_db()
        history = db.execute("""SELECT r.*, c.case_code, c.patient_code, c.predicted_label, u.full_name
                                FROM reports r JOIN cases c ON c.id = r.case_id
                                LEFT JOIN users u ON u.id = r.generated_by
                                ORDER BY r.generated_at DESC, r.id DESC LIMIT 200""").fetchall()
        cases = [case_dict(r) for r in db.execute("SELECT * FROM cases ORDER BY screened_at DESC, id DESC LIMIT 200")]
        selected = None
        if request.args.get("case"):
            selected = get_case_or_404(request.args["case"])
        return render_template("reports.html", history=history, cases=cases, selected=selected)

    @app.route("/cases/<case_code>/report", methods=["POST"])
    @roles_required("doctor", "admin")
    def generate_report(case_code):
        from services.reports import build_report
        case = get_case_or_404(case_code)
        user = current_user()
        filename = build_report(case, app.config["UPLOAD_ROOT"], app.config["REPORT_ROOT"], REVIEW_STATUSES,
                                f"{user['full_name']} ({ROLE_LABELS[user['role']]})")
        db = get_db()
        cur = db.execute("INSERT INTO reports (case_id, filename, generated_by, generated_at) VALUES (?, ?, ?, ?)",
                         (case["id"], filename, user["id"], now_iso()))
        db.commit()
        log_event("report_generated", case_code, {"report_id": cur.lastrowid, "file": filename})
        return redirect(url_for("download_report", report_id=cur.lastrowid))

    @app.route("/reports/<int:report_id>/download")
    @roles_required("doctor", "admin")
    def download_report(report_id):
        row = get_db().execute("""SELECT r.filename, c.case_code FROM reports r JOIN cases c ON c.id = r.case_id
                                  WHERE r.id = ?""", (report_id,)).fetchone()
        if row is None:
            abort(404)
        log_event("report_downloaded", row["case_code"], {"report_id": report_id})
        return send_from_directory(app.config["REPORT_ROOT"], row["filename"], as_attachment=True,
                                   mimetype="application/pdf")

    # ---------- analytics

    @app.route("/analytics")
    @roles_required("doctor", "admin")
    def analytics():
        db = get_db()
        total = db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        by_grade = {r[0]: r[1] for r in db.execute("SELECT predicted_grade, COUNT(*) FROM cases GROUP BY predicted_grade")}
        by_status = {r[0]: r[1] for r in db.execute("SELECT review_status, COUNT(*) FROM cases GROUP BY review_status")}
        by_quality = {r[0]: r[1] for r in db.execute("SELECT quality_status, COUNT(*) FROM cases GROUP BY quality_status")}
        by_priority = {r[0]: r[1] for r in db.execute("SELECT priority, COUNT(*) FROM cases GROUP BY priority")}
        uncertain = db.execute("SELECT COUNT(*) FROM cases WHERE flags_json LIKE '%\"uncertain\"%'").fetchone()[0]
        assessments = {r[0]: r[1] for r in db.execute(
            "SELECT clinician_assessment, COUNT(*) FROM cases WHERE clinician_assessment IS NOT NULL GROUP BY clinician_assessment")}

        start = date.today() - timedelta(days=13)
        daily = {r[0]: r[1] for r in db.execute(
            "SELECT substr(screened_at, 1, 10) d, COUNT(*) FROM cases WHERE screened_at >= ? GROUP BY d", (start.isoformat(),))}
        trend = [{"date": (start + timedelta(days=i)).isoformat(),
                  "label": (start + timedelta(days=i)).strftime("%d %b"),
                  "count": daily.get((start + timedelta(days=i)).isoformat(), 0)} for i in range(14)]

        return render_template(
            "analytics.html", total=total, uncertain=uncertain, trend=trend, metrics=MODEL_METRICS,
            grade_rows=[(f"{i} · {n}", by_grade.get(i, 0)) for i, n in enumerate(CLASS_NAMES)],
            status_rows=[(label, by_status.get(k, 0)) for k, label in REVIEW_STATUSES.items()],
            quality_rows=[(label, by_quality.get(k, 0)) for k, label in QUALITY_LABELS.items()],
            priority_rows=[(label, by_priority.get(k, 0)) for k, label in PRIORITY_LABELS.items()],
            assessment_rows=[(label, assessments.get(k, 0)) for k, label in ASSESSMENTS.items()],
        )

    # ---------- settings & administration

    @app.route("/settings", methods=["GET", "POST"])
    @login_required
    def settings():
        user = current_user()
        db = get_db()
        if request.method == "POST":
            action = request.form.get("action")
            if action == "profile":
                full_name = request.form.get("full_name", "").strip()
                if not 2 <= len(full_name) <= 80:
                    flash("Full name must be 2-80 characters.", "danger")
                else:
                    db.execute("UPDATE users SET full_name = ?, email = ? WHERE id = ?",
                               (full_name, request.form.get("email", "").strip(), user["id"]))
                    db.commit()
                    log_event("profile_updated")
                    flash("Profile updated.", "success")
            elif action == "password":
                if not check_password_hash(user["password_hash"], request.form.get("current", "")):
                    flash("Current password is incorrect.", "danger")
                else:
                    new = request.form.get("new", "")
                    error = _password_error(new, request.form.get("confirm", ""))
                    if error:
                        flash(error, "danger")
                    else:
                        db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(new), user["id"]))
                        db.commit()
                        log_event("password_changed")
                        flash("Password changed.", "success")
            elif action == "rules":
                if user["role"] != "admin":
                    abort(403)
                changes, error = _parse_rule_settings(request.form)
                if error:
                    flash(error, "danger")
                else:
                    for key, value in changes.items():
                        db.execute("UPDATE settings SET value = ? WHERE key = ?", (value, key))
                    db.commit()
                    log_event("review_rules_updated", details=changes)
                    flash("Review rules updated. They apply to new screenings only.", "success")
            return redirect(url_for("settings"))
        return render_template("settings.html", settings=get_settings(), defaults=DEFAULT_SETTINGS)

    @app.route("/admin/users", methods=["GET", "POST"])
    @roles_required("admin")
    def admin_users():
        db = get_db()
        if request.method == "POST":
            action = request.form.get("action")
            if action == "create":
                username = request.form.get("username", "").strip().lower()
                full_name = request.form.get("full_name", "").strip()
                role = request.form.get("role", "")
                password = request.form.get("password", "")
                error = _validate_new_user(username, full_name, password, password)
                if role not in ROLE_LABELS:
                    error = "Please choose a valid role."
                if error:
                    flash(error, "danger")
                else:
                    try:
                        db.execute("INSERT INTO users (username, full_name, email, password_hash, role, is_active, created_at)"
                                   " VALUES (?, ?, ?, ?, ?, 1, ?)",
                                   (username, full_name, request.form.get("email", "").strip(),
                                    generate_password_hash(password), role, now_iso()))
                        db.commit()
                        log_event("user_created", details={"username": username, "role": role})
                        flash(f"Account '{username}' created.", "success")
                    except sqlite3.IntegrityError:
                        flash("That username is already taken.", "danger")
            elif action == "update":
                uid = request.form.get("user_id", type=int)
                role = request.form.get("role", "")
                active = 1 if request.form.get("is_active") == "1" else 0
                target = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
                if target is None or role not in ROLE_LABELS:
                    abort(400)
                if target["id"] == current_user()["id"] and (role != "admin" or not active):
                    flash("You cannot remove your own administrator access.", "warning")
                else:
                    db.execute("UPDATE users SET role = ?, is_active = ? WHERE id = ?", (role, active, uid))
                    db.commit()
                    log_event("user_updated", details={"username": target["username"], "role": role, "active": bool(active)})
                    flash(f"Account '{target['username']}' updated.", "success")
            return redirect(url_for("admin_users"))
        users = db.execute("SELECT * FROM users ORDER BY is_active, created_at DESC").fetchall()
        messages = db.execute("SELECT * FROM contact_messages ORDER BY id DESC LIMIT 20").fetchall()
        return render_template("users.html", users=users, messages=messages)

    @app.route("/admin/audit")
    @roles_required("admin")
    def audit():
        event = request.args.get("event", "")
        case_q = request.args.get("case", "").strip()
        sql, params = "SELECT * FROM audit_log WHERE 1=1", []
        if event:
            sql += " AND event_type = ?"
            params.append(event)
        if case_q:
            sql += " AND case_code LIKE ?"
            params.append(f"%{case_q}%")
        sql += " ORDER BY id DESC LIMIT 300"
        db = get_db()
        events = db.execute(sql, params).fetchall()
        types = [r[0] for r in db.execute("SELECT DISTINCT event_type FROM audit_log ORDER BY event_type")]
        return render_template("audit.html", events=events, types=types, filters=request.args)


# ---------------------------------------------------------------- validation helpers

def _password_error(password, confirm):
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must be at least 8 characters and include letters and numbers."
    if password != confirm:
        return "Passwords do not match."
    return None


def _validate_new_user(username, full_name, password, confirm):
    if not re.match(r"^[a-z0-9_.]{3,30}$", username):
        return "Username must be 3-30 characters: lowercase letters, digits, dot or underscore."
    if not 2 <= len(full_name) <= 80:
        return "Full name must be 2-80 characters."
    return _password_error(password, confirm)


def _parse_rule_settings(form):
    numeric = {
        "uncertainty_margin": (0.0, 1.0), "min_top_score": (0.0, 1.0),
        "dark_threshold": (0, 255), "bright_threshold": (0, 255),
        "min_content_fraction": (0.0, 1.0), "blur_threshold": (0, 10000),
    }
    changes = {}
    for key, (lo, hi) in numeric.items():
        try:
            value = float(form.get(key, ""))
        except ValueError:
            return None, f"'{key.replace('_', ' ')}' must be a number."
        if not lo <= value <= hi:
            return None, f"'{key.replace('_', ' ')}' must be between {lo} and {hi}."
        changes[key] = str(value)
    if float(changes["dark_threshold"]) >= float(changes["bright_threshold"]):
        return None, "Dark threshold must be lower than the bright threshold."
    grades = sorted({int(x) for x in form.getlist("priority_grades") if x.isdigit() and int(x) < len(CLASS_NAMES)})
    changes["priority_grades"] = ",".join(map(str, grades))
    return changes, None


app = create_app()

if __name__ == "__main__":
    # Load the model at startup so the first analysis is fast.
    from services.predictor import get_model
    print("Loading model... (this can take ~10-20 seconds)")
    get_model()
    print("Model ready. Open http://127.0.0.1:5000")
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", use_reloader=False)
