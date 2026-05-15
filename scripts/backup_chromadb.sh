#!/usr/bin/env bash
# ChromaDB on-demand backup to S3.
# Usage: ./scripts/backup_chromadb.sh [ssh_key_path]
#
# Requires: ssh, aws cli (on EC2)
# Default key: ~/.ssh/chroma_key.pem

set -euo pipefail

SSH_KEY="${1:-$HOME/.ssh/chroma_key.pem}"
EC2_HOST="ubuntu@ec2-3-66-231-225.eu-central-1.compute.amazonaws.com"
S3_BUCKET="data-011337673661"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
BACKUP_NAME="chromadb-backup-${TIMESTAMP}.tar.gz"
REMOTE_TMP="/tmp/${BACKUP_NAME}"
S3_PATH="s3://${S3_BUCKET}/backups/${BACKUP_NAME}"

echo "=== ChromaDB Backup ==="
echo "Timestamp: ${TIMESTAMP}"
echo "SSH key:   ${SSH_KEY}"
echo "EC2 host:  ${EC2_HOST}"
echo ""

if [[ ! -f "${SSH_KEY}" ]]; then
    echo "ERROR: SSH key not found at ${SSH_KEY}"
    echo "Usage: $0 [path/to/key.pem]"
    exit 1
fi

echo ">>> Creating backup tarball on EC2..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${EC2_HOST}" \
    "sudo tar czf ${REMOTE_TMP} -C /mnt chroma && ls -lh ${REMOTE_TMP}"

echo ""
echo ">>> Uploading to ${S3_PATH}..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${EC2_HOST}" \
    "aws s3 cp ${REMOTE_TMP} ${S3_PATH}"

echo ""
echo ">>> Verifying upload..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${EC2_HOST}" \
    "aws s3 ls s3://${S3_BUCKET}/backups/ | tail -5"

echo ""
echo ">>> Cleaning up temp file..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${EC2_HOST}" \
    "sudo rm -f ${REMOTE_TMP}"

echo ""
echo "=== Backup complete: ${S3_PATH} ==="
