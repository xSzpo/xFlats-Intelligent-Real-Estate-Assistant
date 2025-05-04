#!/bin/bash
set -xe

# Update & upgrade Ubuntu packages
apt-get update -y
apt-get upgrade -y

# Install Docker, Git, curl, pip & venv support
apt-get install -y docker.io git curl python3-pip python3-venv unzip
systemctl enable --now docker

# Install AWS CLI v2
apt-get install -y unzip &&
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" &&
  unzip awscliv2.zip &&
  ./aws/install &&
  rm -rf awscliv2.zip aws &&
  aws --version

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
  DEVICE=$(lsblk -nd -o NAME,TYPE | awk '/disk/ && $1!="nvme0n1"{print "/dev/"$1; exit}')
fi

# Wait for the device to appear
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
    volumes:
      - /mnt/chroma:/data
    ports:
      - "8000:8000"
    networks:
      - net
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/data

  property-bot:
    image: 274181059559.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler:latest
    restart: always
    networks:
      - net
    environment:
      - AWS_REGION=eu-central-1
      - CHROMADB_IP=chromadb
EOC

chown ubuntu:ubuntu /home/ubuntu/docker-compose.yml

# Create systemd service for Chroma with ECR auth and pull
cat <<EOT >/etc/systemd/system/chroma.service
[Unit]
Description=ChromaDB Docker Compose Service
Requires=docker.service network.target
After=docker.service network.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=ubuntu
Group=docker
WorkingDirectory=/home/ubuntu

# Authenticate to ECR
ExecStartPre=/bin/sh -c 'aws ecr get-login-password --region eu-central-1 \
  | docker login --username AWS --password-stdin 274181059559.dkr.ecr.eu-central-1.amazonaws.com'
# Pull latest images
ExecStartPre=/usr/bin/docker-compose pull
# Start the Compose stack
ExecStart=/usr/bin/docker-compose up -d
# Teardown on stop
ExecStop=/usr/bin/docker-compose down

[Install]
WantedBy=multi-user.target
EOT

# Enable & start the service without aborting on failure
set +e
systemctl daemon-reload
systemctl enable chroma.service
systemctl start chroma.service
set -e
