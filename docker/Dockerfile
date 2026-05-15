FROM python:3.11-slim

# Install cron + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install app
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Create log file
RUN touch /var/log/cron.log

# Install crontab
COPY docker/crontab /etc/cron.d/property-bot-cron
RUN chmod 0644 /etc/cron.d/property-bot-cron

# Run cron in foreground
CMD ["cron", "-f"]
