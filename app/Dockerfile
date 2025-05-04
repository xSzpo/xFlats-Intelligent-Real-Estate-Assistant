FROM python:3.10-slim

# 1) Install cron
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron && \
    rm -rf /var/lib/apt/lists/*

# 2) Install your app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./

# 3) Create the log file (so cron can write to it)
RUN touch /var/log/cron.log

# 4) Drop a system‐cron file into /etc/cron.d
#    - it must include the 'root' user field
#    - use the absolute path to python3
RUN echo "*/30 * * * * root /usr/local/bin/python3 /app/main.py >> /var/log/cron.log 2>&1" \
    > /etc/cron.d/property-bot-cron && \
    chmod 0644 /etc/cron.d/property-bot-cron

# 5) Run cron in the foreground
CMD ["cron", "-f"]
