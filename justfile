# Run 'just' in terminal to few available commands
set default-list

# Lint and format
[group('check')]
lint:
    uv run ruff check --fix
    uv run ruff format

# Check types
[group('check')]
typing:
    uv run ty check src

# Check coverage (pytest)
[group('check')]
cov *args:
    uv run pytest {{ args }}
    uvx --with "setuptools<=80" coverage-badge -fo docs/coverage.svg

# Run all tests on all dependency combinations (nox)
[group('check')]
test *args:
    uv run nox {{ args }}

# Check docstrings on tests
[group('check')]
check-testdocs:
    uv run interrogate tests

# Run linting, formatting, tests and type-checking
[group('check')]
check-all: lint cov test typing check-testdocs

# Build documentation
[group('docs')]
build-docs:
    uv run --group docs sphinx-build docs/source docs/build

# Bring environment up-to-date
[group('lifecycle')]
install:
    uv sync
    uv run prek install --prepare-hooks --overwrite

# Reset environment and all cache files
[group('lifecycle')]
clean:
    uvx pyclean . -d all
    uvx prek uninstall --no-progress
    uvx python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
    uvx python -c "import shutil; shutil.rmtree('.nox', ignore_errors=True)"
    uvx python -c "import shutil; shutil.rmtree('docs/build', ignore_errors=True)"

# Setup environment from scratch
[group('lifecycle')]
fresh: clean install

# Upgrade python and all dependencies
[group('lifecycle')]
upgrade:
    uv sync --upgrade
    uv run prek auto-update --no-progress
