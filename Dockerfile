# ============================
# Build stage
# ============================
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy requirements.txt
COPY ./app/requirements.txt /app/requirements.txt

# Install build dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libsndfile1 \
        ffmpeg \
    && pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc gfortran \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# Copy the app source code
COPY ./app /app

# ============================
# Runtime stage
# ============================
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy app files from builder
COPY --from=builder /app /app

# Copy Firebase credentials (manually included)
COPY ./serviceAccount.json /app/serviceAccount.json

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/serviceAccount.json

# Expose Flask default port
EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

