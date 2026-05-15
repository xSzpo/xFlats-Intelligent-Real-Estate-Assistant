---
name: implement-github-issue
description: End-to-end GitHub issue implementation for xFlats. From issue fetch to plan doc to implementation to PR. Use when user says "implement issue", "work on issue", or provides a GitHub issue number.
---

# Skill: implement-github-issue

End-to-end GitHub issue implementation for xFlats. From issue fetch to merged PR.

## Trigger

User says "implement issue #N", "work on #N", or provides a GitHub issue URL/number.

## Workflow

### Phase 1 — Fetch & Understand

1. Fetch issue: `gh issue view <N> --repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant`
2. Read issue title, description, labels, comments
3. If issue unclear or missing acceptance criteria — list questions, ask user

### Phase 2 — Git Safeguard

1. Check `git status` — must be clean
2. `git checkout main && git pull`
3. Create branch using naming convention (see **Branch & PR Naming** below)

### Phase 3 — Gather Context

1. Read `AGENTS.md` for tech stack + safety rules
2. Read `README.md` for project overview
3. Explore relevant source files in `app/` and/or `infra/`
4. Identify all files that need changes

### Phase 4 — Plan Doc

1. Create plan at `docs/plans/issue-<N>-plan.md` using TEMPLATE.md
2. Present plan to user. Wait for approval before implementing.

### Phase 5 — Implement

1. Work through plan items sequentially
2. Update plan doc status table as items complete
3. Test changes: run `python main.py` (dry run) or verify Docker build

### Phase 6 — PR

1. `git add` changed files
2. Commit: `#<N> <type>: <description>` — see **Branch & PR Naming** for type list
3. Push: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_priv" git push -u origin <branch>`
4. Create PR: `gh pr create --repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant --title "#<N> <type>: <description>" --body "<body>"`
5. Link issue in PR body: `Closes #<N>`

## Branch & PR Naming

### Branch name

| Pattern | When |
|---------|------|
| `<type>/<issue-number>/<short-description>` | Linked to a GitHub issue |
| `<type>/<short-description>` | No GitHub issue |

Examples: `feat/9/google-style-guide`, `fix/12/empty-stats-guard`, `chore/update-deps`

### Allowed types (Conventional Commits)

| Type | Meaning |
|------|---------|
| `feat` | New feature or user-facing capability |
| `fix` | Bug fix |
| `chore` | Maintenance, deps, config — no behavior change |
| `refactor` | Code restructure — no behavior change, no bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `ci` | CI/CD pipeline changes |
| `perf` | Performance improvement |
| `style` | Formatting, whitespace — no logic change |
| `build` | Build system or external dependency changes |

### PR title

| Pattern | When |
|---------|------|
| `#<N> <type>: <short description>` | Linked to a GitHub issue |
| `<type>: <short description>` | No GitHub issue |

Examples: `#9 feat: add Google style guide`, `#12 fix: empty stats guard`, `chore: update deps`

### Commit message

Same format as PR title: `#<N> <type>: <description>` (or `<type>: <description>` if no issue).

## Verification

- Python files: no syntax errors (`python -m py_compile <file>`)
- Docker: `docker build -t xflats .` (ask before running)
- Terraform: `terraform init && terraform plan` (never apply)

## Pitfalls

1. **Private repo** — always use `github.com-priv` SSH host or `--repo` flag with `gh`
2. **Never commit secrets** — API keys go in AWS Secrets Manager
3. **Never `terraform apply`** — plan only
4. **ChromaDB on EC2** — changes to DB schema affect live vector store
5. **Gemini API** — structured extraction prompts in `utils.py` are sensitive to format changes
6. **Cron schedule** — Docker cron runs every 30min, test timing changes carefully
