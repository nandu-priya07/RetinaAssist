# Generic container image for RetinaAssist (builds from this folder).
#   docker build -t retinaassist .
#   docker run -p 7860:7860 -e RETINAASSIST_SECRET_KEY=change-me retinaassist
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    RETINAASSIST_ENV=production \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Fail the build if the trained model is missing or differs from the verified file.
RUN echo "48bec19cb6a312e82f844f20258d9cf7a82ab26dc1b84a461e36777bd835810a  models/retinaassist_efficientnet_finetuned.keras" | sha256sum -c -

USER user
EXPOSE 7860
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 wsgi:app
