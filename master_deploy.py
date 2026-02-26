import subprocess
import sys
import os

print("=== PHASE 1: firmament ===")
subprocess.run([sys.executable, os.path.join("firmament", "firmament.py")])

print("=== PHASE 2: dot reconstruction ===")
subprocess.run([sys.executable, os.path.join("dot_reconstruction", "main.py")])

print("=== DONE ===")
