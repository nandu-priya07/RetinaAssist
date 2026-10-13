# Hugging Face Space (Docker SDK) for RetinaAssist.
# Copy this file and the README.md next to it into the root of the Space repository.
# The source code and the trained model are fetched from GitHub at build time.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    RETINAASSIST_ENV=production \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user

ARG REPO=https://github.com/nandu-priya07/RetinaAssist.git
ARG REF=main
RUN git clone --depth 1 --branch ${REF} ${REPO} /home/user/app && chown -R user:user /home/user/app
WORKDIR /home/user/app

RUN pip install --no-cache-dir -r requirements.txt
# Fail the build if the trained model is missing or differs from the verified file.
RUN echo "48bec19cb6a312e82f844f20258d9cf7a82ab26dc1b84a461e36777bd835810a  models/retinaassist_efficientnet_finetuned.keras" | sha256sum -c -

USER user
EXPOSE 7860
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 wsgi:app
