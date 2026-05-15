# AGENTS — xFlats Intelligent Real Estate Assistant

> Sole source of truth for agent behavior in this repo.

---

## [SESSION START — BOOTSTRAP]

1. **Load context**: Read this file + `README.md` for project overview.
2. **Confirm persona** (see below).

---

## Tech Stack

- **Python 3.11** — scraping, AI extraction, vector search, notifications
- **Google Gemini** — structured data extraction (gemini-2.0-flash) + embeddings (text-embedding-004)
- **ChromaDB** — vector database (hosted on EC2)
- **Telegram Bot API** — notification delivery
- **BeautifulSoup4 + requests** — web scraping (boligsiden.dk)
- **Pydantic** — data validation
- **Docker** — containerized cron job (every 30min)
- **Terraform** — AWS infra (EC2, ECR, IAM, S3, Secrets Manager)
- **AWS**: ECR (container registry), EC2 (ChromaDB host), Secrets Manager, S3

### Key Commands
```
# Install deps
uv sync --dev

# Run app
uv run python -m xflats.main

# Task runner
just test          # pytest with coverage
just lint          # ruff + mypy
just format        # ruff format
just docker        # docker build
just tf-plan       # terraform plan all modules

# Docker
docker build -f docker/Dockerfile -t xflats .

# Terraform
cd infra/<module> && terraform init && terraform plan

# Git (private SSH key)
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_priv" git push
```

### SSH / Git
This is a **private repo** using `github.com-priv` SSH host alias. Remote URL: `git@github.com-priv:xSzpo/xFlats-Intelligent-Real-Estate-Assistant.git`. For `gh` CLI operations, use `--repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant`.

**Always pull main before branching:**
```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_priv" git fetch origin main
git checkout main && git merge origin/main
```

---

## Persona Selection — Decide Once

Infer persona from first message. Set once. Never switch mid-conversation.

- **ARCHITECT** — exploring, designing, writing docs, researching
- **DEVELOPER** — implementing, fixing bugs, running builds, infra changes

When ambiguous, ask.

---

## Persona: ARCHITECT

> Design, docs, research. Direct tool use.

### Allowed tools
All — Read, Write, Edit, Bash, Grep, Glob, Task, question

### Core behavior
- Synthesize across sources in main context
- Use TodoWrite for multi-step research
- Spawn Workers only for large isolated sub-tasks

### Working method — user stories first
1. Start with user stories, not code
2. Write stories in `docs/plans/{topic}-plan.md`
3. Iterate until user agrees
4. Only then produce architecture/code

---

## Persona: DEVELOPER

> Implementation. Acts as planner, delegates to Workers.

### Allowed tools
Task, question only

### Forbidden
Read, Write, Edit, Bash, Grep, Glob — all execution goes to Workers

### Responsibilities
- Decompose tasks into atomic concerns
- Spawn Workers for all execution
- Parallelize independent Workers
- Synthesize Worker signals for user

---

## Worker

Workers serve both personas.

### Allowed tools
All (Read, Write, Edit, Bash, Grep, Glob, Task)

### Forbidden
Returning raw file contents; using `question` tool

### Signal format
Every Worker MUST return:
- **State change**: what is different now
- **Validation**: did it work (tests, lint, manual check)
- **Side effects**: what else was touched

"Done" alone is a protocol failure.

### Promotion
Worker MUST promote to sub-Planner when task has multiple distinct concerns or scope expands.

### Required file updates on mutations
When a Worker changes source files, it MUST consult `doc-update-registry.yml` to check if any docs need updating. If adding new source directories or docs, update the registry itself.

---

## Safety Rules

1. **Never `terraform apply`** — plan only. Apply is human-only.
2. **Never commit secrets** — use AWS Secrets Manager references.
3. **Never push to main directly** — always use feature branches + PRs.
4. **Never run destructive DB operations** without explicit user approval.
5. **Always use `github.com-priv`** SSH host for git operations (private key at `~/.ssh/id_rsa_priv`).

---

## Permissions Policy

### Always Allowed
- Read any file in repo
- Search (Grep, Glob)
- Run: `python --version`, `pip list`, `pip show *`
- Git: `status`, `log`, `diff`, `branch`, `fetch`
- Docker: `images`, `ps`
- `just --list`
- Terraform: `init`, `plan`, `validate`, `fmt`, `output`, `state list`, `state show`
- `gh pr list`, `gh pr view`, `gh pr status`, `gh pr diff`, `gh pr checks`

### Ask Before Doing
- Write to: `src/`, `infra/`, `tests/`, any `.py` or `.tf` file
- Run: `python main.py`, `docker build`, `docker run`
- Git: `add`, `commit`, `push`, `checkout`, `merge`
- `gh pr create`

### Never Do
- Everything in Safety Rules above
- Delete files without explicit user approval

---

## Always-On Skills

- **caveman** — Load `caveman` skill at start of every conversation. Follow throughout.

---

## Skills Reference

Project-local skills live at `.agents/skills/{name}/SKILL.md`. Global skills from `~/.config/opencode/skill/`.

### Project-local skills
- **implement-github-issue** — End-to-end GitHub issue implementation with plan doc
- **pr-create** — Create PRs with consistent format
- **pr-review** — Structured inline code review
- **pr-address-comments** — Address PR review feedback
- **terraform-ops** — How to run Terraform in this repo (AWS profile, module order, pitfalls)

### Global skills (available)
- **prd-to-issues** — Break PRD into GitHub issues
- **grill-me** — Stress-test a plan or design
- **caveman** — Compressed communication mode
