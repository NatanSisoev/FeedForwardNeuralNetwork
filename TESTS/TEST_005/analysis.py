import os
import glob

ROOT_DIR = "TESTS/TEST_005/OUT"

def read_avg(file_path):
    """Return the tag line and average of all numbers except the last line."""
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
    tagged_avgs = []
    baseline_avgs = []

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_avg(filepath)
        if not result:
            continue
        tag, avg = result
        if tag == "X":
            baseline_avgs.append(avg)
        else:
            tagged_avgs.append(avg)

    if not tagged_avgs or not baseline_avgs:
        print("Missing tagged or baseline files.")
        return

    # Compute overall averages
    avg_tagged = sum(tagged_avgs) / len(tagged_avgs)
    avg_baseline = sum(baseline_avgs) / len(baseline_avgs)

    speedup = avg_baseline / avg_tagged

    print(f"Tagged avg: {avg_tagged:.10f}")
    print(f"Baseline avg: {avg_baseline:.10f}")
    print(f"Speedup: {speedup:.6f}")

if __name__ == "__main__":
    main()
