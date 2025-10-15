import os
import glob
from collections import defaultdict

ROOT_DIR = "TESTS/TEST_004/OUT"

def read_avg(file_path):
    """Return the tag and average of all numbers except the last line."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        if len(lines) < 3:
            return None
        tag = lines[1].strip()
        try:
            numbers = [float(x) for x in lines[2:-1]]  # skip last line
        except ValueError:
            return None
        avg = sum(numbers) / len(numbers)
        return tag, avg

def main():
    tag_avgs = {}
    no_tag_avgs = {}

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_avg(filepath)
        if not result:
            continue
        tag, avg = result
        if tag.startswith("NO_"):
            base_tag = tag[3:]  # remove NO_ prefix
            no_tag_avgs[base_tag] = avg
        else:
            tag_avgs[tag] = avg

    # Compute speedups
    speedups = []
    for tag, avg in tag_avgs.items():
        if tag not in no_tag_avgs:
            continue  # skip if no corresponding no_tag
        speedup = no_tag_avgs[tag] / avg
        speedups.append((tag, speedup))

    # Sort by best speedup descending
    speedups.sort(key=lambda x: x[1], reverse=True)

    print("Tag speedups (best first):")
    for tag, sp in speedups:
        print(f"{tag} -> {sp:.6f}")

if __name__ == "__main__":
    main()
