# Skill: pr-create

Create GitHub PRs with consistent format for xFlats.

## Trigger

User says "create PR", "open PR", "submit PR", or task is ready for review.

## Workflow

1. `git status` — confirm clean, on feature branch
2. `git log main..HEAD --oneline` — collect all commits
3. `git diff main...HEAD --stat` — summarize changes
4. Create PR using TEMPLATE.md format:

```bash
gh pr create \
  --repo xSzpo/xFlats-Intelligent-Real-Estate-Assistant \
  --title "<type>: <short description>" \
  --body "$(cat <<'EOF'
<filled template>
EOF
)"
```

## Rules

- Title: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
- Body: use TEMPLATE.md structure
- Always link related issue: `Closes #N` or `Related to #N`
- Push first: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_priv" git push`
