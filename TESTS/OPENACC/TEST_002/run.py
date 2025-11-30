import os
import sys
import subprocess
import re
import string
import time
import yaml
import statistics
from collections import defaultdict

# ---------------- SETTINGS ----------------
SLEEP_INTERVAL = 10  # seconds
DEBUG = True

TEST_NUM = "002"
ROOT_DIR = f"/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/TESTS/OPENACC/TEST_{TEST_NUM}"
REPEAT = "1"
SUBFOLDER = sys.argv[1] if len(sys.argv) > 1 else ""

# ---------------- CREATE NEW SUBFOLDER ----------------
if SUBFOLDER == "":
    os.makedirs(ROOT_DIR, exist_ok=True)
    used = [d for d in os.listdir(f"{ROOT_DIR}/OUT") if d in string.ascii_uppercase]
    next_letter = chr(ord(max(used)) + 1) if used else "A"
    SUBFOLDER = next_letter
    os.makedirs(os.path.join(f"{ROOT_DIR}/OUT", next_letter))

OUTPUT_DIR = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE = f"{ROOT_DIR}/results.md"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCHEDULER_FILE = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/scheduler.sub"

# ---------------- WHAT TO DO ----------------
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

# ---------------- EXECUTION ----------------
if "e" in parts:

    print(f"Saving output to '{OUTPUT_DIR}' and results to '{RESULTS_FILE}'.")

    # Define sequential and parallel flags
    FLAGS_SEQUENTIAL = ""
    FLAGS_PARALLEL = "FEED_INPUT,FORWARD_PROP,BACK_PROP,UPDATE_WEIGHTS"

    all_flags = [("SEQ", FLAGS_SEQUENTIAL), ("PAR", FLAGS_PARALLEL)]
    job_ids = []

    for label, flags in all_flags:

        # Run scheduler
        res = subprocess.run(
            ["sbatch", SCHEDULER_FILE, TEST_NUM, flags, REPEAT, SUBFOLDER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=os.path.dirname(SCHEDULER_FILE)
        )

        job_id = re.search(r"Submitted batch job (\d+)", res.stdout.strip()).group(1)
        job_ids.append((label, job_id))

        if DEBUG:
            print(f"Submitted {label} job {job_id} with flags '{flags}'.")

    # Wait for jobs
    pending_jobs = [jid for _, jid in job_ids]
    wait_time = 0
    while pending_jobs:
        res = subprocess.run(
            ["squeue", "-h", "-j", ",".join(pending_jobs), "-o", "%i", "-u", os.environ["USER"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        running_jobs = set(res.stdout.strip().split("\n")) if res.stdout.strip() else set()
        pending_jobs = [jid for jid in pending_jobs if jid in running_jobs]

        time.sleep(SLEEP_INTERVAL)
        wait_time += SLEEP_INTERVAL
        if DEBUG:
            print(f"{wait_time:3} s\t{len(pending_jobs):3} jobs remaining")

# ---------------- ANALYSIS ----------------
if "a" in parts:

    stats = defaultdict(list)

    # READ OUTPUT FILES
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".out"):
            continue
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "r") as f:
            content = f.read()

        # METADATA
        meta_match = re.search(r"---\s*(.*?)\s*---", content, re.DOTALL)
        if not meta_match:
            continue
        metadata = yaml.safe_load(meta_match.group(1))
        flags = metadata.get("compilation_flags", "")
        if flags:
            flags = flags.replace("-D", "")

        # EXTRACT TIMES
        entries = re.findall(r"#START:\d+\s*(\d+)\s+(\d+\.\d+)", content)
        for acc, t in entries:
            stats[flags].append((int(acc), float(t)))

    # COMPUTE FINAL STATISTICS
    final = {}
    for flags, values in stats.items():
        accs = [v[0] for v in values]
        times = [v[1] for v in values]

        avg_t = sum(times) / len(times)
        std_t = statistics.stdev(times) if len(times) > 1 else 0.0
        min_t = min(times)
        max_t = max(times)

        avg_a = sum(accs) / len(accs)
        std_a = statistics.stdev(accs) if len(accs) > 1 else 0.0
        min_a = min(accs)
        max_a = max(accs)

        final[flags] = (avg_t, std_t, min_t, max_t,
                        avg_a, std_a, min_a, max_a,
                        len(values))

    # Compute speedup
    seq_key = ""  # sequential
    par_key = "FEED_INPUT,FORWARD_PROP,BACK_PROP,UPDATE_WEIGHTS"  # parallel

    speedup = 0.0
    if seq_key in final and par_key in final:
        t_seq = final[seq_key][0]
        t_par = final[par_key][0]
        speedup = t_seq / t_par if t_par > 0 else 0.0

    # WRITE RESULTS
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n# {SUBFOLDER} - Sequential vs Parallel\n```\n")
        f.write(f"{'Version':<15}{'Avg Time(s)':>15}{'Speedup':>10}\n")
        for key in [seq_key, par_key]:
            t_avg = final[key][0] if key in final else 0.0
            spd = speedup if key == par_key else 1.0
            f.write(f"{key:<15}{t_avg:15.6f}{spd:10.4f}\n")
        f.write("```\n")
