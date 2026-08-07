FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CELLVISION_HOST=0.0.0.0 \
    CELLVISION_PORT=8777 \
    CELLVISION_ALLOW_REMOTE=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY review-ui ./review-ui
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8777

CMD ["cellvision", "review-project", "--manifest", "/data/projects/active/project.json", "--host", "0.0.0.0", "--port", "8777", "--allow-remote"]

