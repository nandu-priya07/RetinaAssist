"""Production entry point: gunicorn wsgi:app

Loads the trained model once at startup so the first analysis is fast.
Local development keeps using `python app.py`.
"""
from app import app
from services.predictor import get_model

get_model()
