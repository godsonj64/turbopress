# Control-plane image. Build from the REPO ROOT as context so it can install the
# local turbopress package (for turbopress.certificate) without torch:
#   flyctl deploy . --config service/fly.toml --dockerfile service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install turbopress WITHOUT its deps (no torch): the control plane only needs
# turbopress.certificate, which is pure-Python + pynacl.
COPY pyproject.toml README.md ./
COPY turbopress ./turbopress
RUN pip install --no-cache-dir --no-deps .

# Control-plane dependencies.
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY service ./service

EXPOSE 8000
CMD ["sh", "-c", "uvicorn service.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
