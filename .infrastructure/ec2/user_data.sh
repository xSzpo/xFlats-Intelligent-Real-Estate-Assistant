#!/bin/bash
set -xe

# 1) Update & upgrade Ubuntu packages
apt-get update -y
apt-get upgrade -y

# 2) Install Docker, Git, curl, pip & venv support
apt-get install -y docker.io git curl python3-pip python3-venv
systemctl enable --now docker

# 3) Add ubuntu user to docker group
usermod -aG docker ubuntu

# 4) Install Docker Compose (latest)
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose

# 5) Determine the data volume device path
if ls /dev/nvme1n1 &>/dev/null; then
  DEVICE=/dev/nvme1n1
elif ls /dev/xvdf &>/dev/null; then
  DEVICE=/dev/xvdf
else
  # fallback: first non-root disk
  DEVICE=$(lsblk -nd -o NAME,TYPE | awk '/disk/ && $1!="nvme0n1"{print "/dev/"$1; exit}')
fi

# 6) Wait for that device to appear
while [ ! -e "${DEVICE}" ]; do sleep 1; done

# 7) Format, mount, persist
mkfs -t ext4 "${DEVICE}"
mkdir -p /mnt/chroma
mount "${DEVICE}" /mnt/chroma
echo "${DEVICE} /mnt/chroma ext4 defaults,nofail 0 2" >>/etc/fstab
chown ubuntu:ubuntu /mnt/chroma

# 8) Write Chroma env file
cat <<EOT >/home/ubuntu/.env
CHROMA_SERVER_VERSION=1.0.5
EOT
chown ubuntu:ubuntu /home/ubuntu/.env

# 9) Write docker-compose.yml
cat <<-'EOC' >/home/ubuntu/docker-compose.yml
version: "3.9"

services:
  server:
    image: ghcr.io/chroma-core/chroma:1.0.5
    ports:
      - "8000:8000"
    volumes:
      - /mnt/chroma:/index_data
    restart: always
    env_file:
      - /home/ubuntu/.env
    networks:
      - net

networks:
  net:
    driver: bridge

volumes:
  index_data:
  backups:
EOC
chown ubuntu:ubuntu /home/ubuntu/docker-compose.yml

# 10) Create systemd service for Chroma
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

# 11) Enable & start the service
systemctl daemon-reload
systemctl enable chroma.service
systemctl start chroma.service
