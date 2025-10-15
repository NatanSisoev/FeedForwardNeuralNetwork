import os
import glob
from collections import defaultdict

ROOT_DIR = "TESTS/TEST_004/OUT"

def read_file_avg(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        if len(lines) < 3:
            return None
        tag = lines[1].strip()
        # take all lines from line 2 to the second-to-last line
        try:
            numbers = [float(x) for x in lines[2:-1]]
        except ValueError:
            return None
        avg = sum(numbers) / len(numbers)
        return tag, avg

def main():
    results = []
    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        data = read_file_avg(filepath)
        if data:
            results.append(data)

    if not results:
        print("No valid data found.")
        return

    results.sort(key=lambda x: x[1])

    print("Tag averages:")
    for tag, avg in results:
        print(f"{tag if tag != 'X' else None} -> {avg:e}")

if __name__ == "__main__":
    main()
