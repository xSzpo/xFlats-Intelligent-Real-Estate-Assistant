---
name: terraform-ops
description: How to run Terraform in this repo (AWS profile, module order, pitfalls). Use when working with infra/ modules or user mentions terraform.
---

# Skill: terraform-ops

Run Terraform in this repo. All modules under `infra/`.

## Key Facts

- **AWS Profile**: `priv` — must prefix all commands with `AWS_PROFILE=priv`
- **Region**: `eu-central-1`
- **State bucket**: `terraform-011337673661` (S3, encrypted)
- **State keys**: `state/<module>.tfstate` (e.g. `state/iam.tfstate`)
- **Never `terraform apply`** — plan only. Human applies.

## Module Dependency Order

```
iam  (standalone — deploy first)
 ├── s3
 ├── ecr
 ├── secrets
 └── ec2
```

All modules except `iam` read remote state from `iam` outputs.

## Commands

```bash
# Init (first time or after backend change)
cd infra/<module>
AWS_PROFILE=priv terraform init -reconfigure

# Plan
AWS_PROFILE=priv terraform plan

# Validate only
AWS_PROFILE=priv terraform validate

# Format
terraform fmt
```

## Pitfalls

1. **Always `AWS_PROFILE=priv`** — backend S3 doesn't inherit provider profile
2. **Never apply** — agent plans only, human applies
3. **No variables.tf** — everything hardcoded or from data sources
4. **Remote state coupling** — changing `iam` outputs can break all other modules
5. **EC2 user_data** — changes force instance replacement (destructive)
6. **Security group** — currently allows `0.0.0.0/0` on ports 22 and 8000
