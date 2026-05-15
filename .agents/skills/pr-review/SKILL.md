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

**Python code** (`src/xflats/**/*.py`, `tests/**/*.py`):
- **Google Python Style Guide consistency** — always check:
  - Google-style docstrings on all public functions/classes ([3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings))
  - Full type annotations on args + return ([2.21](https://google.github.io/styleguide/pyguide.html#221-type-annotated-code))
  - `logging` module instead of `print()` ([2.9](https://google.github.io/styleguide/pyguide.html#29-deprecated-language-features))
  - Specific exception types, not bare `except Exception` ([2.4](https://google.github.io/styleguide/pyguide.html#24-exceptions))
  - `# TODO(username): description` format ([3.12](https://google.github.io/styleguide/pyguide.html#312-todo-comments))
  - snake_case / CamelCase naming conventions ([3.16](https://google.github.io/styleguide/pyguide.html#316-naming))
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

**Docker** (`docker/Dockerfile`):
- Base image pinned?
- Cron schedule correct?
- Layer ordering efficient?

### 3. Check Doc-Update Registry

**Always check `doc-update-registry.yml` against the PR diff.**

1. Read `doc-update-registry.yml`
2. For each changed file in the PR, match against `source` globs
3. For each match, check if the linked `docs` were also updated in the PR
4. If a linked doc was NOT updated — flag as **Medium** finding:
   ```
   **[Medium]** `doc-update-registry.yml` requires `<doc>` to be reviewed when `<source>` changes. Either update the doc or confirm no change needed.
   ```
5. If a linked doc WAS updated — verify the doc content is consistent with the source changes. Flag inaccuracies as **Medium**.

### 4. Post Review to GitHub

**Always post findings as GitHub PR review comments — both inline and summary.**

#### 4a. Inline comments

Use `gh api` to submit a PR review with inline comments on specific files/lines:

```bash
gh api repos/xSzpo/xFlats-Intelligent-Real-Estate-Assistant/pulls/<N>/reviews \
  --method POST \
  --field event=COMMENT \
  --field body="<summary>" \
  --field 'comments=[{"path":"<file>","line":<line>,"body":"<comment>"}]'
```

Each inline comment body format:
```
**[SEVERITY]** Problem description. Fix: suggestion.
```

Severity levels:
- **Critical** — blocks merge (security, data loss, broken functionality)
- **Medium** — should fix (bugs, bad patterns)
- **Low** — nice to fix (style, minor improvements)
- **Nit** — optional (formatting, naming)

Tips:
- `line` must be within the diff hunk (use `gh pr diff` to find valid line numbers)
- For lines not in the diff, include them in the summary comment instead
- Group related findings into one inline comment when on adjacent lines
- Use `side=RIGHT` for lines in the new version of the file

#### 4b. Summary comment

The review body (passed as `--field body=`) should contain:
1. Overall verdict: **Approve** / **Request Changes** / **Comment**
2. Counts by severity
3. Any findings that couldn't be attached inline (lines outside diff)

### 5. Also show findings to user

After posting to GitHub, display a concise summary in the conversation so user sees findings without leaving the terminal.
