# The control plane itself.
#
# Note what this image does *not* contain: a Docker daemon. The platform talks
# to the host's daemon over the mounted socket, so this is a small Python
# image and not a docker-in-docker stack.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, from the project metadata alone, so the layer is reused
# whenever only the source changes -- which is every deploy.
COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.30" "docker>=7.1" \
      "httpx>=0.27" "pydantic>=2.8"

COPY src /app/src
COPY web /app/web

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    NANOFAAS_ROOT=/app

EXPOSE 8000

# Runs as root, deliberately and only here: the process needs access to the
# mounted Docker socket, and dropping to a non-root user that is then added to
# the docker group buys nothing -- membership of that group is already root
# equivalent. The function containers this starts do run as non-root; see
# images.py. The trade is documented in docker-compose.yml.
CMD ["uvicorn", "nanofaas.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
