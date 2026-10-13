"""End-to-end tests for RetinaAssist using the REAL saved model.

Run from the project folder:   python -m unittest discover -s tests -v
"""
import io
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import warnings
from contextlib import contextmanager

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import create_app  # noqa: E402
from services import gradcam, predictor, priority, quality  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "test_images")
SAMPLES = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))) if os.path.isdir(SAMPLE_DIR) else []
PASSWORD = "TestPass123"


def sample_bytes():
    if not SAMPLES:
        raise unittest.SkipTest("Put a retinal image in test_images/ to run this test.")
    with open(os.path.join(SAMPLE_DIR, SAMPLES[0]), "rb") as f:
        return f.read(), SAMPLES[0]


def png_bytes(color, size=(400, 300)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


class ServiceTests(unittest.TestCase):
    def test_model_shapes(self):
        model = predictor.get_model()
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        self.assertEqual(model.output_shape, (None, 5))

    def test_preprocess_keeps_0_255_range(self):
        data, name = sample_bytes()
        batch = predictor.preprocess(predictor.load_image(data, name))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertGreater(batch.max(), 1.0)  # not divided by 255
        self.assertLessEqual(batch.max(), 255.0)

    def test_real_prediction(self):
        data, name = sample_bytes()
        result = predictor.predict(predictor.load_image(data, name))
        self.assertEqual(len(result["scores"]), 5)
        self.assertAlmostEqual(sum(result["scores"]), 1.0, places=3)
        self.assertEqual(result["grade"], int(np.argmax(result["scores"])))
        self.assertEqual(result["label"], predictor.CLASS_NAMES[result["grade"]])

    def test_invalid_files_rejected(self):
        data, _ = sample_bytes()
        bad = [(b"", "x.png"), (b"garbage" * 50, "x.png"), (data[:4000], "x.png"),
               (data, "x.gif"), (png_bytes("red", (50, 50)), "tiny.png")]
        buf = io.BytesIO(); Image.new("RGB", (300, 300)).save(buf, "GIF")
        bad.append((buf.getvalue(), "disguised.png"))  # GIF content with .png name
        for content, name in bad:
            with self.subTest(name=name), self.assertRaises(predictor.ImageValidationError):
                predictor.load_image(content, name)

    def test_gradcam_outputs(self):
        data, name = sample_bytes()
        img = predictor.load_image(data, name)
        cam = gradcam.compute_heatmap(img, predictor.predict(img)["grade"])
        self.assertEqual(cam.ndim, 2)
        self.assertGreaterEqual(cam.min(), 0.0)
        self.assertLessEqual(cam.max(), 1.0)
        original, heat, overlay = gradcam.render(img, cam)
        self.assertEqual(original.size, heat.size)
        self.assertEqual(original.size, overlay.size)

    def test_quality_checks(self):
        self.assertEqual(quality.check_quality(Image.new("RGB", (400, 400), "black"))["status"], "fail")
        self.assertEqual(quality.check_quality(Image.new("RGB", (400, 400), (250, 250, 250)))["status"], "fail")
        data, name = sample_bytes()
        self.assertIn(quality.check_quality(predictor.load_image(data, name))["status"], ("pass", "warning"))

    def test_priority_rules(self):
        settings = {"uncertainty_margin": "0.15", "min_top_score": "0.5", "priority_grades": "3,4"}
        ok_q = {"status": "pass", "checks": []}
        confident = {"grade": 0, "label": "No DR", "top_score": 0.9, "second_grade": 1, "second_score": 0.05, "margin": 0.85}
        self.assertEqual(priority.evaluate(confident, ok_q, settings), ("routine", []))
        close = dict(confident, top_score=0.45, second_score=0.40, margin=0.05)
        prio, flags = priority.evaluate(close, ok_q, settings)
        self.assertEqual(prio, "medium")
        self.assertEqual(flags[0]["code"], "uncertain")
        severe = dict(confident, grade=3, label="Severe DR")
        prio, flags = priority.evaluate(severe, ok_q, settings)
        self.assertEqual((prio, flags[0]["code"]), ("high", "priority_grade"))
        bad_q = {"status": "fail", "checks": [{"status": "fail", "message": "Too dark."}]}
        prio, flags = priority.evaluate(confident, bad_q, settings)
        self.assertEqual((prio, flags[0]["code"]), ("high", "recapture"))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Flask's test client keeps served files open until garbage collection; not a real leak.
        warnings.simplefilter("ignore", ResourceWarning)
        cls.tmp = tempfile.mkdtemp()
        cls.app = create_app({
            "TESTING": True, "SECRET_KEY": "test", "CSRF_ENABLED": False, "DEMO_PASSWORD": PASSWORD,
            "DATABASE": os.path.join(cls.tmp, "test.db"),
            "UPLOAD_ROOT": os.path.join(cls.tmp, "uploads"), "REPORT_ROOT": os.path.join(cls.tmp, "reports"),
        })
        os.makedirs(cls.app.config["UPLOAD_ROOT"], exist_ok=True)
        os.makedirs(cls.app.config["REPORT_ROOT"], exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def login(self, username):
        client = self.app.test_client()
        r = client.post("/login", data={"username": username, "password": PASSWORD})
        self.assertEqual(r.status_code, 302, f"login failed for {username}")
        return client

    def upload(self, client, content, name, code="SYN-0001"):
        return client.post("/screening", data={"patient_code": code, "image": (io.BytesIO(content), name)},
                           content_type="multipart/form-data")

    @contextmanager
    def db(self):
        conn = sqlite3.connect(self.app.config["DATABASE"])
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def test_public_pages(self):
        c = self.app.test_client()
        for path in ["/", "/about", "/contact", "/login", "/register"]:
            with self.subTest(path=path):
                r = c.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertIn(b"prototype", r.data.lower())

    def test_protected_routes_redirect(self):
        c = self.app.test_client()
        for path in ["/dashboard", "/screening", "/records", "/reports", "/analytics", "/settings", "/admin/users", "/admin/audit"]:
            with self.subTest(path=path):
                r = c.get(path)
                self.assertEqual(r.status_code, 302)
                self.assertIn("/login", r.headers["Location"])

    def test_bad_login_and_password_hashing(self):
        r = self.app.test_client().post("/login", data={"username": "doctor", "password": "wrong"})
        self.assertEqual(r.status_code, 401)
        with self.db() as conn:
            h = conn.execute("SELECT password_hash FROM users WHERE username='doctor'").fetchone()[0]
        self.assertNotIn(PASSWORD, h)

    def test_role_restrictions(self):
        staff = self.login("staff")
        for path in ["/reports", "/analytics", "/admin/users", "/admin/audit"]:
            with self.subTest(role="staff", path=path):
                self.assertEqual(staff.get(path).status_code, 403)
        admin = self.login("admin")
        self.assertEqual(admin.get("/screening").status_code, 403)
        doctor = self.login("doctor")
        self.assertEqual(doctor.get("/admin/users").status_code, 403)
        for path in ["/dashboard", "/screening", "/records", "/reports", "/analytics", "/settings"]:
            with self.subTest(role="doctor", path=path):
                self.assertEqual(doctor.get(path).status_code, 200)
        for path in ["/admin/users", "/admin/audit", "/analytics", "/reports"]:
            with self.subTest(role="admin", path=path):
                self.assertEqual(admin.get(path).status_code, 200)

    def test_full_screening_review_report_flow(self):
        data, name = sample_bytes()
        staff = self.login("staff")
        r = self.upload(staff, data, name, "SYN-1001")
        self.assertEqual(r.status_code, 302)
        case_code = r.headers["Location"].rsplit("/", 1)[-1]

        with self.db() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_code = ?", (case_code,)).fetchone()
        self.assertIsNotNone(row)
        expected = predictor.predict(predictor.load_image(data, name))
        self.assertEqual(row["predicted_grade"], expected["grade"])  # stored result = real model output
        self.assertIsNone(row["gradcam_error"])

        page = staff.get(f"/cases/{case_code}")
        self.assertEqual(page.status_code, 200)
        self.assertIn(expected["label"].encode(), page.data)
        for n in predictor.CLASS_NAMES:
            self.assertIn(n.encode(), page.data)
        for kind in ["original.png", "heatmap.png", "overlay.png"]:
            img = staff.get(f"/media/{case_code}/{kind}")
            self.assertEqual(img.status_code, 200)
            self.assertEqual(img.mimetype, "image/png")
        self.assertEqual(staff.get(f"/media/{case_code}/../../app.py").status_code, 404)

        # Staff cannot review or create reports
        self.assertEqual(staff.post(f"/cases/{case_code}/review", data={"review_status": "reviewed"}).status_code, 403)
        self.assertEqual(staff.post(f"/cases/{case_code}/report").status_code, 403)

        doctor = self.login("doctor")
        r = doctor.post(f"/cases/{case_code}/review", data={
            "review_status": "reviewed", "clinician_assessment": "disagree", "clinician_grade": "1", "notes": "Test note"})
        self.assertEqual(r.status_code, 302)
        with self.db() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_code = ?", (case_code,)).fetchone()
            events = [e[0] for e in conn.execute("SELECT event_type FROM audit_log WHERE case_code = ?", (case_code,))]
        self.assertEqual((row["review_status"], row["clinician_assessment"], row["clinician_grade"]), ("reviewed", "disagree", 1))
        for ev in ["case_created", "analysis_succeeded", "case_viewed", "review_status_changed", "clinician_disagree", "notes_updated"]:
            self.assertIn(ev, events)

        r = doctor.post(f"/cases/{case_code}/report", follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "application/pdf")
        self.assertTrue(r.data.startswith(b"%PDF"))
        self.assertGreater(len(r.data), 20000)  # contains embedded images
        reports_page = doctor.get("/reports")
        self.assertIn(case_code.encode(), reports_page.data)
        self.assertIn(case_code.encode(), doctor.get(f"/reports?case={case_code}").data)

        self.assertIn(case_code.encode(), doctor.get("/records?q=SYN-1001").data)
        self.assertNotIn(case_code.encode(), doctor.get("/records?q=NOPE-999").data)
        self.assertIn(case_code.encode(), doctor.get("/records?status=reviewed").data)
        analytics = doctor.get("/analytics")
        self.assertEqual(analytics.status_code, 200)
        # Research-model validation figures live on About only, not on dashboard/analytics.
        self.assertNotIn(b"0.595", analytics.data)
        self.assertNotIn(b"79.8", doctor.get("/dashboard").data)
        self.assertIn(b"0.595", self.app.test_client().get("/about").data)

    def test_invalid_uploads(self):
        staff = self.login("staff")
        self.assertEqual(self.upload(staff, b"not an image at all", "fake.png").status_code, 400)
        self.assertEqual(self.upload(staff, b"abc", "doc.pdf").status_code, 400)
        data, name = sample_bytes()
        self.assertEqual(self.upload(staff, data, name, code="bad code!").status_code, 400)
        big = self.upload(staff, b"0" * (11 * 1024 * 1024), "big.png")
        self.assertEqual(big.status_code, 302)  # 413 handler redirects back with a message
        with self.db() as conn:
            failed = conn.execute("SELECT COUNT(*) FROM audit_log WHERE event_type = 'analysis_failed'").fetchone()[0]
        self.assertGreaterEqual(failed, 2)

    def test_dark_image_flags_recapture(self):
        staff = self.login("staff")
        r = self.upload(staff, png_bytes((3, 3, 3)), "dark.png", "SYN-DARK")
        self.assertEqual(r.status_code, 302)
        code = r.headers["Location"].rsplit("/", 1)[-1]
        with self.db() as conn:
            row = conn.execute("SELECT quality_status, flags_json, priority FROM cases WHERE case_code = ?", (code,)).fetchone()
        self.assertEqual(row["quality_status"], "fail")
        self.assertIn("recapture", row["flags_json"])

    def test_registration_requires_approval(self):
        c = self.app.test_client()
        r = c.post("/register", data={"full_name": "New Person", "username": "newperson", "email": "",
                                      "password": "Newpass123", "confirm": "Newpass123"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(c.post("/login", data={"username": "newperson", "password": "Newpass123"}).status_code, 403)
        admin = self.login("admin")
        with self.db() as conn:
            uid = conn.execute("SELECT id FROM users WHERE username='newperson'").fetchone()[0]
        admin.post("/admin/users", data={"action": "update", "user_id": uid, "role": "doctor", "is_active": "1"})
        self.assertEqual(c.post("/login", data={"username": "newperson", "password": "Newpass123"}).status_code, 302)
        self.assertEqual(c.get("/reports").status_code, 200)  # role now doctor, read from DB

    def test_admin_rules_update(self):
        admin = self.login("admin")
        r = admin.post("/settings", data={"action": "rules", "uncertainty_margin": "0.2", "min_top_score": "0.4",
                                          "dark_threshold": "20", "bright_threshold": "235", "min_content_fraction": "0.1",
                                          "blur_threshold": "10", "priority_grades": ["2", "3", "4"]})
        self.assertEqual(r.status_code, 302)
        with self.db() as conn:
            self.assertEqual(conn.execute("SELECT value FROM settings WHERE key='priority_grades'").fetchone()[0], "2,3,4")
        # Non-admins cannot change rules
        self.assertEqual(self.login("doctor").post("/settings", data={"action": "rules"}).status_code, 403)
        admin.post("/settings", data={"action": "rules", "uncertainty_margin": "0.15", "min_top_score": "0.5",
                                      "dark_threshold": "25", "bright_threshold": "230", "min_content_fraction": "0.15",
                                      "blur_threshold": "15", "priority_grades": ["3", "4"]})

    def test_audit_log_is_append_only(self):
        self.app.test_client().get("/")  # ensure db exists
        with self.db() as conn:
            conn.execute("INSERT INTO audit_log (created_at, event_type) VALUES ('now', 'test')")
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("UPDATE audit_log SET event_type = 'x'")
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM audit_log")

    def test_contact_form(self):
        c = self.app.test_client()
        self.assertEqual(c.post("/contact", data={"name": "A", "email": "bad", "message": "x"}).status_code, 400)
        self.assertEqual(c.post("/contact", data={"name": "Ann", "email": "ann@example.org",
                                                  "message": "Interested in the prototype."}).status_code, 302)

    def test_csrf_enforced(self):
        self.app.config["CSRF_ENABLED"] = True
        try:
            r = self.app.test_client().post("/login", data={"username": "doctor", "password": PASSWORD})
            self.assertEqual(r.status_code, 400)
        finally:
            self.app.config["CSRF_ENABLED"] = False


if __name__ == "__main__":
    unittest.main()
