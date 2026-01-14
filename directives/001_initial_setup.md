# Directive 001: Initial Environment Setup

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
# Windows: .venv\Scripts\activate
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

