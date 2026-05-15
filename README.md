# xFlats AI Agent

**An intelligent real-estate assistant for finding apartment deals in Copenhagen and Warsaw via Telegram**

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Development](#development)
8. [CI/CD](#cicd)
9. [Infrastructure](#infrastructure)

---

## Overview

Finding a great apartment in Copenhagen or Warsaw can be a race against time. xFlats AI Agent automates the hunt by:

- Scraping top real-estate sites every 30 minutes (boligsiden.dk for Copenhagen, otodom.pl for Warsaw)
- Supporting multiple regions via a `RegionConfig` data class — each region defines its own site, search URLs, AI models, and notification preferences
- Extracting and structuring listings with Google Gemini
- Storing embeddings in a ChromaDB vector database on AWS EC2
- Alerting you via Telegram when a listing matches your criteria

---

## Features

- **Automated Scraping**
  - Docker cron job runs every 30 minutes on EC2
  - Scrapes boligsiden.dk (Copenhagen) and otodom.pl (Warsaw)
  - Multi-region support via `RegionConfig` (`config/regions.py`)
- **AI-driven Extraction**
  - Gemini 2.0 Flash pulls structured data from HTML
  - Pydantic models validate each offer
- **Vector Search**
  - Embeddings (`text-embedding-004`) stored in ChromaDB
  - Similarity search & price-point analysis against historical data
- **Telegram Notifications**
  - Matching offers sent directly to Telegram group
- **Geocoding & Transit**
  - Enriches listings with coordinates and public transit info

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                   EC2 Instance                       │
│                                                      │
│  ┌────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │ Docker Cron │───▶│  Scrapers   │───▶│  Gemini   │ │
│  │ (30 min)    │    │ boligsiden  │    │ Extraction│ │
│  └────────────┘    │ otodom      │    └─────┬─────┘ │
│                    └─────────────┘          │       │
│                                              │       │
│                                              ▼       │
│                    ┌─────────────┐    ┌───────────┐  │
│                    │  Telegram   │◀───│ ChromaDB  │  │
│                    │  Notifier   │    │ (Vector)  │  │
│                    └──────┬──────┘    └───────────┘  │
│                           │                          │
└───────────────────────────┼──────────────────────────┘
                            ▼
                     Telegram Group
```

**Flow:** Cron triggers per-region scrapers → Gemini extracts structured data → stored in ChromaDB (one collection per region) → new matches sent via Telegram.

---

## Tech Stack

| Layer          | Technology                                      |
|----------------|------------------------------------------------|
| Language       | Python 3.14                                     |
| Package Mgmt   | [uv](https://docs.astral.sh/uv/)              |
| Task Runner    | [just](https://github.com/casey/just)           |
| AI Extraction  | Google Gemini (`gemini-2.0-flash`)              |
| Embeddings     | Google `text-embedding-004`                     |
| Vector DB      | ChromaDB (hosted on EC2)                        |
| Scraping       | BeautifulSoup4 + requests                       |
| Validation     | Pydantic                                        |
| Notifications  | Telegram Bot API                                |
| Secrets        | AWS Secrets Manager                             |
| Infra          | Terraform (EC2, ECR, IAM, S3, Secrets Manager) |
| Container      | Docker (cron job every 30 min)                  |
| CI/CD          | GitHub Actions                                  |
| Linting        | Ruff + pre-commit hooks                         |

---

## Project Structure

```
xFlats-Intelligent-Real-Estate-Assistant/
├── README.md
├── AGENTS.md
├── pyproject.toml              # Python deps + tool config (uv)
├── uv.lock                     # Lockfile
├── justfile                    # Task runner (just)
├── .pre-commit-config.yaml     # Hooks: ruff, detect-secrets
├── .github/workflows/
│   ├── ci.yml                  # Lint + test on PR
│   └── deploy.yml              # Docker build + ECR push
├── src/xflats/
│   ├── main.py                 # Entry point (RealEstateScraper)
│   ├── utils.py                # Shared helpers (geocoding, transit)
│   ├── scraper/boligsiden.py   # Web scraping (Copenhagen)
│   ├── scraper/otodom.py       # Web scraping (Warsaw)
│   ├── extraction/gemini.py    # AI extraction
│   ├── storage/chromadb.py     # Vector DB ops
│   ├── notifications/telegram.py
│   ├── config/secrets.py       # AWS Secrets Manager
│   └── config/regions.py       # Multi-region configuration
├── tests/
│   ├── conftest.py
│   └── unit/                   # Unit tests
├── docker/
│   ├── Dockerfile
│   └── crontab
├── infra/                      # Terraform (ec2, ecr, iam, s3, secrets)
├── notebooks/                  # Jupyter exploration
└── docs/
    └── architecture.md         # System architecture
```

---

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) — package manager
- [just](https://github.com/casey/just) — task runner
- Docker
- Terraform (for infra changes)
- AWS credentials configured

### Setup

```bash
git clone git@github.com-priv:xSzpo/xFlats-Intelligent-Real-Estate-Assistant.git
cd xFlats-Intelligent-Real-Estate-Assistant

uv sync --dev
just install-dev       # installs pre-commit hooks
pre-commit install
```

---

## Development

All common tasks via `just`:

```bash
just test       # Run unit tests
just lint       # Ruff linting
just format     # Ruff formatting
just docker     # Build Docker image
just tf-plan    # Terraform plan (never apply directly)
```

**Pre-commit hooks** run automatically on commit: Ruff linting/formatting and secret detection (`detect-secrets`).

---

## CI/CD

Two GitHub Actions workflows:

| Workflow       | Trigger              | What it does                          |
|----------------|----------------------|---------------------------------------|
| `ci.yml`       | Pull request         | Runs linting (`ruff`) + unit tests    |
| `deploy.yml`   | Push to `main`       | Builds Docker image, pushes to ECR    |

---

## Infrastructure

Terraform modules in `infra/`:

| Module    | Purpose                                        |
|-----------|------------------------------------------------|
| `ec2`     | EC2 instance running Docker + ChromaDB         |
| `ecr`     | Container registry for Docker images           |
| `iam`     | Roles and policies                             |
| `s3`      | Storage buckets                                |
| `secrets` | AWS Secrets Manager (API keys, tokens)         |

> **Never run `terraform apply` directly.** Use `just tf-plan` to review changes, then apply through approved process only.
