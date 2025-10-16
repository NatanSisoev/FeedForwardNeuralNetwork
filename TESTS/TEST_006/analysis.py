import os
import glob
from collections import defaultdict

ROOT_DIR = "TESTS/TEST_006/OUT"

def read_avg(file_path):
    """Return the tag line and average of all timing lines (ignoring last line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        if len(lines) < 3:
            return None
        tag = lines[1].strip()
        try:
            # Convert all numeric lines except the last one to floats
            numbers = [float(x) for x in lines[2:-1]]
        except ValueError:
            return None
        avg = sum(numbers) / len(numbers)
        return tag, avg

def main():
    tag_times = defaultdict(list)

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_avg(filepath)
        if not result:
            continue
        tag, avg = result
        tag_times[tag].append(avg)

    if not tag_times:
        print("No valid .out files found.")
        return

    # Compute average per tag combination
    tag_avg = {tag: sum(vals) / len(vals) for tag, vals in tag_times.items()}

    # Extract baseline
    if "X" not in tag_avg:
        print("No baseline (tag 'X') found.")
        return

    baseline_avg = tag_avg["X"]

    # Compute speedups relative to baseline
    speedups = []
    for tag, avg in tag_avg.items():
        if tag == "X":
            continue
        sp = baseline_avg / avg
        speedups.append((tag, avg, sp))

    # Sort by descending speedup
    speedups.sort(key=lambda x: x[2], reverse=True)

    # Print all averages
    print("Average times by tag:")
    for tag, avg in sorted(tag_avg.items(), key=lambda x: x[1]):
        label = "BASELINE" if tag == "X" else tag
        print(f"{label:60s} -> {avg:.10f}")

    print("\nAverage baseline:", f"{baseline_avg:.10f}")
    print("\nSpeedup ranking (vs baseline):")
    for tag, avg, sp in speedups:
        print(f"{tag:60s} -> speedup {sp:.6f}")

if __name__ == "__main__":
    main()
