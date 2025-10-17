import sys
import subprocess

TEST_NUM            = sys.argv[1]
GENERATION_FILE     = f"TESTS/TEST_{TEST_NUM}/run.py"

subprocess.run(["python3", GENERATION_FILE, *sys.argv[2:]])

