---
name: pr-review
description: Structured inline code review for xFlats PRs. Use when user says "review PR", "code review", or provides a PR URL/number.
---

# Skill: pr-review

Structured inline code review for xFlats PRs.

## Trigger

User says "review PR", "review this PR", "code review", or provides a PR URL/number.

## Workflow

### 1. Identify PR

```bash
gh pr view <N> --repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant
gh pr diff <N> --repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant
```

### 2. Analyze Diff

Review changes by type:

**Python code** (`app/*.py`):
- Type safety (Pydantic models correct?)
- Error handling (API calls, network, parsing)
- Secret handling (no hardcoded keys)
- Gemini prompt changes (structured output format intact?)
- ChromaDB operations (embeddings, queries)

**Terraform** (`infra/**/*.tf`):
- State impact (new resources vs modifying existing)
- Security groups / IAM scope
- Cost implications
- Variable validation

**Docker** (`app/Dockerfile`):
- Base image pinned?
- Cron schedule correct?
- Layer ordering efficient?

### 3. Post Review

Post each finding as one line:

```
**[SEVERITY]** `file:line` — Problem. Fix: suggestion.
```

Severity levels:
- **Critical** — blocks merge (security, data loss, broken functionality)
- **Medium** — should fix (bugs, bad patterns)
- **Low** — nice to fix (style, minor improvements)
- **Nit** — optional (formatting, naming)

### 4. Summary

End with overall assessment: Approve / Request Changes / Comment.
