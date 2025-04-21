#!/bin/bash
set -xe

# Update & upgrade Ubuntu packages
apt-get update -y
apt-get upgrade -y

# Install Docker, Git, curl, pip & venv support
apt-get install -y docker.io git curl python3-pip python3-venv
systemctl enable --now docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install Docker Compose (latest)
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# Determine the data volume device path
if ls /dev/nvme1n1 &>/dev/null; then
  DEVICE=/dev/nvme1n1
elif ls /dev/xvdf &>/dev/null; then
  DEVICE=/dev/xvdf
else
  # fallback: first non-root disk
  DEVICE=$(lsblk -nd -o NAME,TYPE | awk '/disk/ && $1!="nvme0n1"{print "/dev/"$1; exit}')
fi

# Wait for that device to appear
while [ ! -e "${DEVICE}" ]; do sleep 1; done

# Format, mount, persist
mkfs -t ext4 "${DEVICE}"
mkdir -p /mnt/chroma
mount "${DEVICE}" /mnt/chroma
echo "${DEVICE} /mnt/chroma ext4 defaults,nofail 0 2" >>/etc/fstab
chown ubuntu:ubuntu /mnt/chroma

# Write docker-compose.yml
cat <<-'EOC' >/home/ubuntu/docker-compose.yml
networks:
  net:
    driver: bridge

services:
  chromadb:
    image: chromadb/chroma:latest
    restart: always

    # Persist your data volume
    volumes:
      - /mnt/chroma:/data

    # Expose the HTTP port
    ports:
      - "8000:8000"

    networks:
      - net

    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/data
EOC

chown ubuntu:ubuntu /home/ubuntu/docker-compose.yml

# Create systemd service for Chroma
cat <<EOT >/etc/systemd/system/chroma.service
[Unit]
Description=ChromaDB Docker Compose Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0
User=ubuntu

[Install]
WantedBy=multi-user.target
EOT

# Enable & start the service
systemctl daemon-reload
systemctl enable chroma.service
systemctl start chroma.service
