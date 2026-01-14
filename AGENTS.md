# AGENTS.md - System Kernel

## Project Context

**Name:** Basic NPU Speech-to-Text Test Module
**Purpose:** A lightweight, standalone Python script to validate and benchmark NPU-accelerated Whisper transcription on Intel Core Ultra hardware.
**Stack:** Python, OpenVINO GenAI, SoundDevice, Silero VAD

## Core Domain Entities

- VAD: Voice Activity Detection (Silero)
- Inference: Running audio through model to get text
- NPU: Intel Core Ultra Neural Processing Unit
- Real-Time Factor (RTF): Processing time / Audio durationRatio
- Chunk: Discrete audio segment
- OpenVINO IR: Intermediate Representation (.xml + .bin)

---

## 1. The Prime Directive

You are an Anti-Gravity Agent operating on the Basic NPU Speech-to-Text Test Module codebase.

**Before writing ANY code:**
1. Read `docs/PRD.md` to understand WHAT we are building
2. Read `docs/ARCH.md` to understand HOW we structure it
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
