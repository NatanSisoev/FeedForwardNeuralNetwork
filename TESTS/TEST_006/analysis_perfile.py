import os
import glob
from collections import defaultdict

ROOT_DIR = "TESTS/TEST_006/OUT"

def read_avg(file_path):
    """Return the tag line and average of all numeric lines except the last line."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        if len(lines) < 3:
            return None
        tag = lines[1].strip()
        try:
            numbers = [float(x) for x in lines[2:-1]]
        except ValueError:
            return None
        avg = sum(numbers) / len(numbers)
        return tag, avg

def main():
    tag_times = defaultdict(list)
    file_results = []

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_avg(filepath)
        if not result:
            continue
        tag, avg = result
        tag_times[tag].append(avg)
        file_results.append((os.path.basename(filepath), tag, avg))

    if not tag_times:
        print("No valid .out files found.")
        return

    if "X" not in tag_times:
        print("No baseline (tag 'X') found.")
        return

    # Compute overall average per tag
    tag_avg = {tag: sum(vals) / len(vals) for tag, vals in tag_times.items()}
    baseline_avg = tag_avg["X"]

    # Compute per-file speedups
    per_file_speedups = []
    for filename, tag, avg in file_results:
        if tag == "X":
            continue
        sp = baseline_avg / avg
        per_file_speedups.append((filename, tag, avg, sp))

    # Compute overall speedup ranking per tag
    overall_speedups = []
    for tag, avg in tag_avg.items():
        if tag == "X":
            continue
        sp = baseline_avg / avg
        overall_speedups.append((tag, avg, sp))
    overall_speedups.sort(key=lambda x: x[2], reverse=True)

    # Print overall tag averages
    print("Overall average times by tag:")
    for tag, avg in sorted(tag_avg.items(), key=lambda x: x[1]):
        label = "BASELINE" if tag == "X" else tag
        print(f"{label:60s} -> {avg:.10f}")

    print("\nAverage baseline:", f"{baseline_avg:.10f}")

    # Print overall speedup ranking
    print("\nSpeedup ranking (vs baseline):")
    for tag, avg, sp in overall_speedups:
        print(f"{tag:60s} -> speedup {sp:.6f}")

    # Print per-file averages and speedups
    print("\nPer-file averages and speedups:")
    for filename, tag, avg, sp in per_file_speedups:
        print(f"{filename:40s} | {tag:40s} | avg: {avg:.10f} | speedup: {sp:.6f}")

if __name__ == "__main__":
    main()
