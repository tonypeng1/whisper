# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run app: `streamlit run trans_real_time.py`
- Run fine-tuned app: `streamlit run trans_real_time_fine_tuned.py`
- Install dependencies: `pip install -e .`
- Docker build: `docker build -t realtime-transcription-app:1.0 .`
- Docker run: `docker compose up`

## Code Style Guidelines
- Python version: 3.11.1 (strict requirement)
- Type annotations: Use Python type hints for function parameters and return values
- Imports: Group standard library, third-party, and local imports with single blank line between groups
- Docstrings: Google-style docstrings for functions and classes
- Error handling: Use try/except with specific error types and meaningful error messages
- Variable naming: snake_case for variables and functions
- Function parameters: Use underscore prefix for internal function parameters (_param_name)
- Line length: Keep lines under 88 characters
- Indentation: 4 spaces, no tabs
- PyTorch conventions: Use _device for device parameters, move tensors to device with .to(device)