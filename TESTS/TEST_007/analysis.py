import re
import glob
from collections import defaultdict

LOG_FOLDER = "TESTS/TEST_007/OUT/parallel"  # "sequential" or "parallel"

time_re = re.compile(r"(loading_patterns|pattern_shuffle|input_feed|forward_propagation|backward_propagation|weight_update|predicting|recognition):([0-9.]+)")

def parse_file(file_path):
    phase_times = defaultdict(float)
    total_elapsed = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            m = time_re.match(line)
            if m:
                phase, t = m.groups()
                phase_times[phase] += float(t)
            else:
                # Last line with total elapsed
                tmp = line.split("\t")
                if len(tmp) == 2:
                    try:
                        total_elapsed = float(tmp[1])
                    except ValueError:
                        pass

    return phase_times, total_elapsed

def analyze_logs():
    all_phase_times = defaultdict(float)
    totals = []
    file_count = 0

    for file_path in glob.glob(f"{LOG_FOLDER}/*.out"):
        phase_times, total_elapsed = parse_file(file_path)
        if total_elapsed is None:
            continue
        file_count += 1
        totals.append(total_elapsed)
        for phase, t in phase_times.items():
            all_phase_times[phase] += t

    if file_count == 0:
        print("No valid log files found.")
        return

    avg_total = sum(totals) / file_count
    print(f"Processed {file_count} files, average total elapsed = {avg_total:.6f} s\n")

    print("Phase breakdown (average per file):")
    for phase, t in sorted(all_phase_times.items(), key=lambda x: x[1], reverse=True):
        avg_time = t / file_count
        pct = (avg_time / avg_total) * 100
        print(f"{phase:25s}: {avg_time:9.6f} s ({pct:6.2f}%)")

if __name__ == "__main__":
    analyze_logs()
