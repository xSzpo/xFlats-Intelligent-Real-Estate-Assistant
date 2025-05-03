# docker build -t property-bot .
# docker run -d \
# --name property-bot \
# -v ~/.aws:/root/.aws:ro \
# -e AWS_PROFILE=priv \
# property-bot

# Use an official slim Python image
FROM python:3.10-slim

# Install cron
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only .py files from the project root
COPY *.py ./

# Add a cron job: run main.py every 15 minutes and log output
RUN echo "*/15 * * * * root python /app/main.py >> /var/log/cron.log 2>&1" \
    > /etc/cron.d/property-bot-cron && \
    chmod 0644 /etc/cron.d/property-bot-cron && \
    crontab /etc/cron.d/property-bot-cron

# Ensure cron runs in foreground (so container doesn’t exit)
CMD ["cron", "-f"]
