#!/usr/bin/env python3
"""
Verify that the development environment is correctly configured.
run this after initial setup.
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    required = (3, 10)
    current = sys.version_info[:2]
    if current < required:
        return False, f"Python {required[0]}.{required[1]}+ required, found {current[0]}.{current[1]}"
    return True, f"Python {current[0]}.{current[1]} ✓"

def check_env_file():
    env_path = Path(".env")
    if not env_path.exists():
        return False, ".env file not found (copy from .env.example)"
    return True, ".env file exists ✓"

def check_required_dirs():
    required = ['.tmp', 'directives', 'execution', 'src', 'src/stt_npu', 'tests', 'scripts', 'models', 'docs']
    missing = [d for d in required if not Path(d).is_dir()]
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "All directories exist ✓"

def check_dependencies():
    # Check simple imports
    try:
        import numpy
        import sounddevice
        import librosa
        import openvino_genai
    except ImportError as e:
        return False, f"Missing dependency: {e}"
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
        print(f"[{status}] {name}: {message}")
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
