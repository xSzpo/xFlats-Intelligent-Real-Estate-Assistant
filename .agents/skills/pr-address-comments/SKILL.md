# Skill: pr-address-comments

Address PR review feedback on xFlats PRs.

## Trigger

User says "address comments", "fix review comments", "handle PR feedback", or provides a PR with pending review comments.

## Workflow

### 1. Fetch Comments

```bash
gh api repos/xSzpo/xFlats-Intelligent-Real-Estate-Assistant/pulls/<N>/comments
gh api repos/xSzpo/xFlats-Intelligent-Real-Estate-Assistant/pulls/<N>/reviews
```

### 2. Analyze Each Comment

For each comment, classify:

| Verdict | Meaning |
|---------|---------|
| **Accept** | Valid point, implement as suggested |
| **Accept-modified** | Valid point, implement differently (explain why) |
| **Reject** | Disagree — provide reasoning |
| **Already done** | Already addressed in current code |
| **Out of scope** | Valid but belongs in separate issue |

### 3. Present Analysis

Show table to user before implementing:

| # | Comment | Verdict | Action |
|---|---------|---------|--------|
| 1 | ... | Accept | ... |

Wait for user approval.

### 4. Implement

- Work through accepted items
- Commit with message: `fix: address review comments`
- Push to same branch

### 5. Reply on GitHub

Reply to every comment via:

```bash
gh api repos/xSzpo/xFlats-Intelligent-Real-Estate-Assistant/pulls/<N>/comments \
  -f body="<response>" -f in_reply_to=<comment_id>
```

Key: reply endpoint is `POST pulls/{N}/comments` with `in_reply_to` field, NOT `comments/{id}/replies`.
