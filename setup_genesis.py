#!/usr/bin/env python3
"""
Genesis Setup Script
Creates project scaffold based on PRD.md and ARCH.md
"""

import os
import sys
from pathlib import Path

# Configuration extracted/summarized from docs
PROJECT_NAME = "Basic NPU Speech-to-Text Test Module"
PROJECT_DESCRIPTION = "A lightweight, standalone Python script to validate and benchmark NPU-accelerated Whisper transcription on Intel Core Ultra hardware."

# Files to verify existence of (but not embed content of)
PRD_FILE = "docs/PRD.md"  
ARCH_FILE = "docs/ARCH.md"

DIRECTORIES = [
    ".tmp",
    "directives",
    "execution",
    "src",
    "src/stt_npu",
    "tests",
    "scripts",
    "models",
    "docs" # Ensure docs exists
]

DEPENDENCIES = [
    "openvino-genai>=2024.5",
    "sounddevice",
    "numpy",
    "librosa",
    "silero-vad",
    "torch", # Required by Silero
    "onnxruntime", # Likely needed for Silero ONNX
    "setuptools",
    "pytest",
    "python-dotenv"
]

ENTITIES = [
    "VAD: Voice Activity Detection (Silero)",
    "Inference: Running audio through model to get text",
    "NPU: Intel Core Ultra Neural Processing Unit",
    "Real-Time Factor (RTF): Processing time / Audio durationRatio",
    "Chunk: Discrete audio segment",
    "OpenVINO IR: Intermediate Representation (.xml + .bin)"
]

def check_docs_exist():
    """Verify that the planning documents exist."""
    print(f"Checking for {PRD_FILE} and {ARCH_FILE}...")
    if not Path(PRD_FILE).exists():
        print(f"WARNING: {PRD_FILE} not found. Some references may be broken.")
    if not Path(ARCH_FILE).exists():
        print(f"WARNING: {ARCH_FILE} not found. Some references may be broken.")

def create_directories():
    print("Creating directory structure...")
    for directory in DIRECTORIES:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py for python packages
        if directory.startswith("src") and "stt_npu" in directory:
             (Path(directory) / "__init__.py").touch()
    
    # Create src/__init__.py
    (Path("src") / "__init__.py").touch()
    
    # Create tests mirror structure
    (Path("tests") / "__init__.py").touch()
    
    print("Directories created.")

def create_gitignore():
    print("Creating .gitignore...")
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/
*.egg-info/
dist/
build/

# Environment
.env
.env.local
*.local

# IDE
.idea/
.vscode/
*.swp
*.swo
.cursorrules

# Project
.tmp/
*.log
models/
__pycache__/
"""
    Path(".gitignore").write_text(content)

def create_env_example():
    print("Creating .env.example...")
    content = """# NPU Configuration
# No specific secrets needed for local NPU inference yet
# Include device overrides if necessary
STT_DEVICE=NPU
"""
    Path(".env.example").write_text(content)

def create_readme():
    print("Creating README.md...")
    # Dynamic reading of ARCH dict structure would go here normally, 
    # but we will use the scaffolded list for the readme view
    tree = "\n".join([f"- {d}/" for d in DIRECTORIES])
    
    content = f"""# {PROJECT_NAME}

{PROJECT_DESCRIPTION}

## Quick Start

1. Clone the repository
2. Copy `.env.example` to `.env` and configure
3. Run `python setup_genesis.py` (if not already run)
4. Follow `directives/001_initial_setup.md`

## Documentation

- [Product Requirements]({PRD_FILE})
- [Technical Architecture]({ARCH_FILE})
- [Agent Instructions](AGENTS.md)

## Project Structure

```
{tree}
```
"""
    Path("README.md").write_text(content, encoding="utf-8")

def create_requirements():
    print("Creating requirements.txt...")
    content = "\n".join(DEPENDENCIES)
    Path("requirements.txt").write_text(content, encoding="utf-8")

def create_agents_md():
    print("Creating AGENTS.md...")
    entities_list = "\n".join([f"- {e}" for e in ENTITIES])
    
    content = f"""# AGENTS.md - System Kernel

## Project Context

**Name:** {PROJECT_NAME}
**Purpose:** {PROJECT_DESCRIPTION}
**Stack:** Python, OpenVINO GenAI, SoundDevice, Silero VAD

## Core Domain Entities

{entities_list}

---

## 1. The Prime Directive

You are an Anti-Gravity Agent operating on the {PROJECT_NAME} codebase.

**Before writing ANY code:**
1. Read `{PRD_FILE}` to understand WHAT we are building
2. Read `{ARCH_FILE}` to understand HOW we structure it
3. Check `directives/` for your current assignment

**Core Rules:**
- Use ONLY the technologies defined in ARCH.md Tech Stack
- Use ONLY the terms defined in ARCH.md Dictionary
- Follow ONLY the API contracts defined in ARCH.md
- Place code ONLY in the directories specified in ARCH.md

---

## 2. The 3-Layer Workflow

### Layer 1: Directives (Orders)
- Location: `directives/`
- Purpose: Task assignments with specific acceptance criteria
- Action: Read the lowest-numbered incomplete directive

### Layer 2: Orchestration (Planning)
- Location: `.tmp/`
- Purpose: Your scratchpad for planning and notes
- Action: Break complex tasks into steps before coding

### Layer 3: Execution (Automation)
- Location: `execution/`
- Purpose: Reusable scripts for repetitive tasks
- Examples: `run_setup.py`, `scripts/test_npu.py`

---

## 3. The Testing Mandate

**No implementation without a failing test.**

Workflow:
1. Write test in `tests/` that describes expected behavior
2. Run test, confirm it fails
3. Write minimum code in `src/` to make test pass
4. Refactor if needed
5. Confirm all tests still pass

---

## 4. Definition of Done

A task is complete when:
- [ ] Code exists in appropriate `src/` subdirectory
- [ ] All new code has corresponding tests in `tests/`
- [ ] All tests pass (`pytest`)
- [ ] Type checking passes
- [ ] Linting passes
- [ ] Related PRD User Story acceptance criteria are met
- [ ] Directive file is marked as Complete

---

## 5. File Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `core.py` |
| Python classes | PascalCase | `class Transcriber` |
| Test files | `test_` prefix | `test_core.py` |
| Directives | `NNN_description.md` | `001_initial_setup.md` |

---

## 6. Commit Message Format
```
type(scope): description

[optional body]

Refs: directive-NNN
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
"""
    Path("AGENTS.md").write_text(content, encoding="utf-8")

def create_initial_directive():
    print("Creating directives/001_initial_setup.md...")
    content = """# Directive 001: Initial Environment Setup

## Objective

Configure the development environment and verify all dependencies are working.

## Prerequisites

- Python 3.10+ installed
- Intel NPU Driver installed (on Host)
- OpenVINO Toolkit compatible hardware

## Steps

### Step 1: Virtual Environment
```bash
python -m venv .venv
# Activate:
# Windows: .venv\\Scripts\\activate
# Linux/Mac: source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
```bash
cp .env.example .env
# Verify STT_DEVICE=NPU
```

### Step 4: Verify Setup
```bash
python execution/verify_setup.py
```

### Step 5: Run Initial Tests
```bash
pytest tests/ -v
```

## Acceptance Criteria

- [ ] Virtual environment created and activated
- [ ] All dependencies (OpenVINO, SoundDevice, etc.) installed
- [ ] `.env` file exists
- [ ] `verify_setup.py` passes all checks
- [ ] `pytest` runs

## Status: [ ] Incomplete / [ ] Complete

## Notes

"""
    (Path("directives") / "001_initial_setup.md").write_text(content, encoding="utf-8")

def create_verify_script():
    print("Creating execution/verify_setup.py...")
    content = f"""#!/usr/bin/env python3
\"\"\"
Verify that the development environment is correctly configured.
run this after initial setup.
\"\"\"

import sys
import importlib
from pathlib import Path

def check_python_version():
    required = (3, 10)
    current = sys.version_info[:2]
    if current < required:
        return False, f"Python {{required[0]}}.{{required[1]}}+ required, found {{current[0]}}.{{current[1]}}"
    return True, f"Python {{current[0]}}.{{current[1]}} ✓"

def check_env_file():
    env_path = Path(".env")
    if not env_path.exists():
        return False, ".env file not found (copy from .env.example)"
    return True, ".env file exists ✓"

def check_required_dirs():
    required = {DIRECTORIES}
    missing = [d for d in required if not Path(d).is_dir()]
    if missing:
        return False, f"Missing directories: {{', '.join(missing)}}"
    return True, "All directories exist ✓"

def check_dependencies():
    # Check simple imports
    try:
        import numpy
        import sounddevice
        import librosa
        import openvino_genai
    except ImportError as e:
        return False, f"Missing dependency: {{e}}"
    return True, "Critical dependencies importable ✓"

def main():
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Directory Structure", check_required_dirs),
        ("Dependencies Import", check_dependencies),
    ]
    
    print("=" * 50)
    print("Environment Verification")
    print("=" * 50)
    
    all_passed = True
    for name, check_func in checks:
        passed, message = check_func()
        status = "✓" if passed else "✗"
        print(f"[{{status}}] {{name}}: {{message}}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("All checks passed! Environment is ready.")
        return 0
    else:
        print("Some checks failed. Be sure to activate .venv and pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
    (Path("execution") / "verify_setup.py").write_text(content, encoding="utf-8")

def create_ide_config():
    print("Creating .cursorrules...")
    content = f"""# Cursor AI Rules for {PROJECT_NAME}

## Session Start Protocol
ALWAYS read these files at the start of EVERY session:
1. AGENTS.md (this project's conventions and workflow)
2. {ARCH_FILE} (technical architecture and constraints)
3. directives/ (find your current task)

## Code Generation Rules
- Use ONLY technologies listed in ARCH.md Tech Stack
- Follow directory structure defined in ARCH.md (src/stt_npu)
- Use domain terms EXACTLY as defined in ARCH.md Dictionary
- Write tests BEFORE implementation

## Forbidden Actions
- Do NOT install packages not listed in requirements.txt without approval
- Do NOT create files outside the defined directory structure
- Do NOT deviate from API contracts in ARCH.md
- Do NOT use .tmp/ for anything except temporary planning notes
"""
    Path(".cursorrules").write_text(content, encoding="utf-8")

def main():
    print(f"Initializing {PROJECT_NAME}...")
    check_docs_exist()
    create_directories()
    create_gitignore()
    create_env_example()
    create_readme()
    create_requirements()
    create_agents_md()
    create_initial_directive()
    create_verify_script()
    create_ide_config()
    print("Genesis complete! Run: python execution/verify_setup.py")

if __name__ == "__main__":
    main()
