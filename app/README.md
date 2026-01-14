```bash
AWS_PROFILE=priv CHROMADB_IP=3.75.94.164 NUMBER_OF_PAGES_TO_OPEN=1 python -m main
```

```bash

AWS_PROFILE=priv \
aws ecr get-login-password \
  --region eu-central-1 \
| docker login \
    --username AWS \
    --password-stdin 011337673661.dkr.ecr.eu-central-1.amazonaws.com

docker buildx create --use

# grab the current Git commit short hash
AWS_PROFILE=priv GIT_HASH=$(git rev-parse --short HEAD) docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -t 011337673661.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler-pl:latest \
  -t 011337673661.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler-pl:${GIT_HASH} \
  .

docker pull --platform linux/amd64 011337673661.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler-pl:latest

docker run --platform linux/amd64 -d -e CHROMADB_IP='3.75.94.164' --name property-bot 011337673661.dkr.ecr.eu-central-1.amazonaws.com/xflats-crawler-pl:latest

```


```bash
AWS_PROFILE=priv CHROMADB_IP=3.75.94.164 NUMBER_OF_PAGES_TO_OPEN=15 python -m main

```