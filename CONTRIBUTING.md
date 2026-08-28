# Contributing to Smart AI Studio

Thank you for your interest in contributing to Smart AI Studio!

## Code of Conduct
We are committed to providing a friendly, safe, and welcoming environment for all contributors.

## Development Setup
1. Fork and clone the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the test suite:
   ```bash
   python3 -m pytest tests/ -v
   ```

## Pull Request Guidelines
- Ensure all 80+ unit and integration tests pass before submitting PRs.
- Write tests for any new features or bug fixes.
- Follow Python PEP 8 style standards and high-readability naming conventions.
- All contributions are subject to the project's [Commercial Source-Available License](LICENSE).
