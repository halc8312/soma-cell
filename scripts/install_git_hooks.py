#!/usr/bin/env python3
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[1]
subprocess.check_call(["git", "config", "core.hooksPath", ".githooks"], cwd=root)
print("Installed SOMA repository hooks: core.hooksPath=.githooks")
