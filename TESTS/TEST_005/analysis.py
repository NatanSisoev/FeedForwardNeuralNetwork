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
    tagged_avg = None
    baseline_avg = None

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_avg(filepath)
        if not result:
            continue
        tag, avg = result
        if tag == "X":
            baseline_avg = avg
        else:
            tagged_avg = avg

    if tagged_avg is None or baseline_avg is None:
        print("Missing tagged or baseline file.")
        return

    speedup = baseline_avg / tagged_avg

    print(f"Tagged avg: {tagged_avg:.10f}")
    print(f"Baseline avg: {baseline_avg:.10f}")
    print(f"Speedup: {speedup:.6f}")

if __name__ == "__main__":
    main()
