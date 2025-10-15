import os
import glob
from collections import defaultdict

ROOT_DIR = "TESTS/TEST_004/OUT"

def read_out_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        if len(lines) < 3:
            return None
        try:
            tags = lines[1].strip()
            data_lines = [line.strip().split("\t") for line in lines[2:] if "\t" in line]
            encerts = [int(x[0]) for x in data_lines]
            times = [float(x[1]) for x in data_lines]
        except ValueError:
            return None

        if any(e != 885 for e in encerts):
            return None

        avg_time = sum(times) / len(times)
        return tags, avg_time

def main():
    tag_times = defaultdict(list)

    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        result = read_out_file(filepath)
        if result:
            tags, avg_time = result
            tag_times[tags].append(avg_time)

    if not tag_times:
        print("No valid data found.")
        return

    tag_avg = {tag: sum(v) / len(v) for tag, v in tag_times.items()}
    sorted_tags = sorted(tag_avg.items(), key=lambda x: x[1])

    print("Average times by tag:")
    for tag, t in sorted_tags:
        print(f"{tag if tag != 'X' else None} -> {t}")

if __name__ == "__main__":
    main()
