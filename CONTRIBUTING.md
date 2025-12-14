# Contributing to ML Toolkit

Thank you for your interest in contributing to ML Toolkit! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/ml-toolkit.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests to ensure everything works
6. Commit your changes with clear commit messages
7. Push to your fork and submit a pull request

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- We use [Flake8](https://flake8.pycqa.org/) for linting
- We use [MyPy](http://mypy-lang.org/) for type checking
- All code should follow PEP 8 guidelines

Run formatting and linting before submitting:
```bash
black src/
flake8 src/
mypy src/
```

## Testing

- Write tests for all new features and bug fixes
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage

```bash
pytest tests/ -v
pytest tests/ --cov=src/ml_toolkit
```

## Documentation

- Update README.md if you add new features
- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Ensure all tests pass and code is properly formatted
3. Update documentation as needed
4. The PR will be merged once you have approval from maintainers

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

## Questions?

Feel free to open an issue for any questions or concerns!
