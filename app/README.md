```bash
AWS_PROFILE=priv CHROMADB_IP=3.124.214.10 python -m main
```

```bash

AWS_PROFILE=priv \
aws ecr get-login-password \
  --region eu-central-1 \
| docker login \
    --username AWS \
    --password-stdin 274181059559.dkr.ecr.eu-central-1.amazonaws.com

docker buildx create --use

# grab the current Git commit short hash
GIT_HASH=$(git rev-parse --short HEAD) \
AWS_PROFILE=priv \
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t 274181059559.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler:latest \
  -t 274181059559.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler:${GIT_HASH} \
  .

docker pull --platform linux/amd64 274181059559.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler:latest

docker run --platform linux/amd64 -d -e CHROMADB_IP='dddd' --name property-bot 274181059559.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler:latest

```


```bash
AWS_PROFIL=priv CHROMADB_IP=3.124.214.10 NUMBER_OF_PAGES_TO_OPEN=15 python -m main

```