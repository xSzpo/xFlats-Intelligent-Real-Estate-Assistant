#!/bin/bash
set -xe

amazon-linux-extras enable docker
yum clean metadata
yum install -y docker git
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# Install Python 3.10 from source
cd /usr/src
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar xzf Python-3.10.13.tgz
cd Python-3.10.13
./configure --enable-optimizations
make altinstall

# Set Python 3.10 as default
ln -sf /usr/local/bin/python3.10 /usr/bin/python3
ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

# Verify Python version
python3 --version
pip3 --version

# Install Rust (required for chromadb client)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
echo 'source $HOME/.cargo/env' >>/home/ec2-user/.bashrc

# Upgrade pip and install chromadb
pip3 install --upgrade pip setuptools wheel
pip3 install chromadb

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

# Wait for EBS volume
while [ ! -e /dev/xvdf ]; do sleep 1; done
mkfs -t ext4 /dev/xvdf
mkdir -p /mnt/chroma
mount /dev/xvdf /mnt/chroma
echo "/dev/xvdf /mnt/chroma ext4 defaults,nofail 0 2" >>/etc/fstab
chown ec2-user:ec2-user /mnt/chroma

# .env for Chroma version
cat <<EOT >/home/ec2-user/.env
CHROMA_SERVER_VERSION=1.0.5
EOT
chown ec2-user:ec2-user /home/ec2-user/.env

# docker-compose.yml with proper indentation
cat <<-'EOC' >/home/ec2-user/docker-compose.yml
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
      - /home/ec2-user/.env
    networks:
      - net

networks:
  net:
    driver: bridge

volumes:
  index_data:
  backups:
EOC

chown ec2-user:ec2-user /home/ec2-user/docker-compose.yml

# systemd service to run docker-compose
cat <<EOT >/etc/systemd/system/chroma.service
[Unit]
Description=ChromaDB Docker Compose Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0
User=ec2-user

[Install]
WantedBy=multi-user.target
EOT

# Enable the service
systemctl daemon-reexec
systemctl daemon-reload
systemctl enable chroma.service
systemctl start chroma.service
