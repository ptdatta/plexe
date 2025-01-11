# smolmodels
✨ build specialised ml models from natural language

## 1. Setup Instructions
This project uses `poetry` for dependency management. To set up your development environment, follow these steps:

### 1.1. Install Poetry
Install `poetry` using `pip`:
```bash
pip install poetry
```
Alternatively, you can follow the [official Poetry installation guide](https://python-poetry.org/docs/#installation)
for other methods.

### 1.2. [Optional] Configure Poetry
If you prefer to create the virtual environment inside the project directory, run this command:
```bash
poetry config virtualenvs.in-project true
```
This step is optional. By default, `poetry` manages virtual environments centrally in `~/.cache/pypoetry/virtualenvs`.

### 1.3. Install Dependencies
Run the `setup.py` script to install all dependencies and initialise the pre-commit hooks:
```bash
python setup.py
```
We use a Python setup script to be platform-agnostic.

### 1.4. [Optional] Activate Virtual Environment
To activate the virtual environment manually, run:
```bash
poetry shell
```
This step is optional as `poetry run` or `python setup.py` will handle the environment automatically.
