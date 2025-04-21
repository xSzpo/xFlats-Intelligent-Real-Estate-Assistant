# xFlats AI Agent

**An intelligent real-estate assistant for finding apartment deals in Copenhagen via Telegram**

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Tech Stack](#tech-stack)  
5. [Initial Proof‑of‑Concept](#initial-proof‑of‑concept)  
6. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Setup](#setup)  
7. [Usage](#usage)  
8. [Future Work](#future-work)  
9. [References](#references)  

---

## Overview

Finding a great apartment in Copenhagen can be a race against time. xFlats AI Agent automates the hunt by:

- Crawling top Danish real‑estate sites every 30 minutes  
- Extracting and structuring listings with Google’s Gemini API  
- Storing embeddings in a ChromaDB vector database on AWS EC2  
- Alerting you via Telegram when a listing matches your criteria  
- Letting you interact with the bot in a dedicated Telegram group

---

## Features

- **Automated Crawling**  
  - Scheduled every 30 minutes via AWS CloudWatch → Lambda  
  - Respects `robots.txt` before scraping  
- **AI‑driven Extraction**  
  - Gemini “JSON mode” pulls structured data from HTML  
  - Pydantic models validate each offer  
- **Vector Search & RAG**  
  - Embeddings (text-embedding-004) stored in ChromaDB  
  - Similarity search & price-point analysis against historical data  
- **Event‑driven Notifications**  
  - Matching offers → SQS queue → Lambda → Telegram group  
- **Interactive Telegram Agent**  
  - Monitors group chat  
  - Responds to on‑demand queries (e.g., “Show me similar apartments”)  

---

## Architecture

```text
┌──────────────┐    CloudWatch    ┌──────────────┐    SQS     ┌──────────────┐
│  Scheduler   │  ────────────▶   │  Crawler     │  ────────▶ │  Notifier    │
│(30 min cron) │                  │(AWS Lambda)  │            │(AWS Lambda)  │
└──────────────┘                  └──────────────┘            └──────────────┘
       │                                    │                        │
       │                                    ▼                        │
       │                           ┌─────────────────┐               │
       │                           │ ChromaDB Vector │◀──────────────┘
       │                           │ Database (EC2)  │
       │                           └─────────────────┘
       │                                    ▲
       │                                    │
       └────────────────────────────────────┘
                     Telegram
                     Group & Bot
