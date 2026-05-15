# xFlats — project task runner
# Usage: just <recipe>

set shell := ["bash", "-cu"]

# List all available recipes
default:
    @just --list

# Run pytest with coverage
test:
    uv run pytest tests/ --cov=src/xflats --cov-report=term-missing

# Run ruff check + mypy
lint:
    uv run ruff check src/ tests/
    uv run mypy src/

# Run ruff format
format:
    uv run ruff format src/ tests/

# Build Docker image
docker:
    docker build -f docker/Dockerfile -t xflats .

# Terraform plan for all infra modules
tf-plan:
    for dir in infra/*/; do echo "=== $dir ===" && terraform -chdir=$dir plan; done

# Remove caches and build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf .ruff_cache .mypy_cache .pytest_cache dist *.egg-info

# Install dependencies
install:
    uv sync

# Install with dev dependencies
install-dev:
    uv sync --dev

# Install pre-commit hooks
pre-commit:
    uv run pre-commit install
