# Contributing

Thank you for your interest in improving this repository!

## Getting Started

```bash
git clone https://github.com/avuppal/llm-finetuning
cd llm-finetuning
pip install -r requirements.txt
pip install pytest pytest-cov ruff black
```

## Running Tests

```bash
# All CPU tests (required before any PR)
pytest -m "not gpu" -v

# With coverage
pytest -m "not gpu" --cov=src --cov-report=term-missing
```

All tests must pass on CPU without downloading model weights.
GPU tests (marked `@pytest.mark.gpu`) are opt-in.

## Code Style

This project uses:
- **[ruff](https://github.com/astral-sh/ruff)** for linting
- **[black](https://github.com/psf/black)** for formatting (line length 100)

```bash
ruff check src/ tests/
black src/ tests/ --line-length 100
```

## Pull Request Guidelines

1. **Branch** from `main`: `git checkout -b feat/your-feature`
2. **Write tests** for new functionality (CPU-only, mocked)
3. **Run tests** — `pytest -m "not gpu"` must pass
4. **Update docs** — if you change the training pipeline, update `docs/architecture.md`
5. **Update CHANGELOG.md** under `[Unreleased]`
6. **Open PR** against `main` with a clear description

## Adding a New Dataset Format

1. Add a formatter function `format_<name>(example) -> (prompt, completion)` in `src/dataset.py`
2. Register it in `_get_formatter()` 
3. Add tests in `tests/test_dataset.py`

## Adding a New Evaluation Metric

1. Implement in `src/evaluate.py`
2. Add unit tests with known inputs/outputs (no model needed)
3. Wire into `run_evaluation()` and the CLI

## Reporting Issues

Please include:
- Python version
- CUDA / GPU info (if applicable)
- Minimal reproducer
- Expected vs actual behaviour
