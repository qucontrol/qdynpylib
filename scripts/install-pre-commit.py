#!/usr/bin/env python
"""Install the pre-commit hooks

This should be run from the Makefile immediately after creating .venv/py37
"""
import os


if os.path.isdir(".git") and not os.path.isfile(".git/hooks/pre-commit"):
    print("##################################################################")
    print("Installing pre-commit hooks")
    print("##################################################################")
    os.system("tox -e run-cmd -- pre-commit install")
