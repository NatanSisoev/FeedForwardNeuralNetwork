import os
import sys
import subprocess
import itertools
import re
import string
import time
import yaml
import statistics
from collections import defaultdict

# SETTINGS
SLEEP_INTERVAL = 10  # seconds
DEBUG          = True

# SCHEDULER ARGUMENTS
TEST_NUM    = "001"
ROOT_DIR    = f"TESTS/TEST_{TEST_NUM}"
FLAGS_RAW   = ""  # determined later
REPEAT      = "1"
SUBFOLDER   = sys.argv[1] if len(sys.argv) > 1 else ""

# CREATE NEW SUBFOLDER (IF NONE INDICATED)
if SUBFOLDER == "":
    os.makedirs(ROOT_DIR, exist_ok=True)
    used = [d for d in os.listdir(f"{ROOT_DIR}/OUT") if d in string.ascii_uppercase]
    next_letter = chr(ord(max(used)) + 1) if used else "A"
    SUBFOLDER = next_letter
    os.makedirs(os.path.join(f"{ROOT_DIR}/OUT", next_letter))

# FILE PATHS
SCHEDULER_FILE   = "scheduler.sub"
OUTPUT_DIR       = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE     = f"{ROOT_DIR}/results.md"

# WHAT TO DO: execute (e), analyze (a)
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

# EXECUTION
if "e" in parts:
    # FOLDER INFO
    print(f"Saving output to '{OUTPUT_DIR}' and results to '{RESULTS_FILE}'.")

    # ALL FLAGS
    flags = [
        "FEED_INPUT",
        "TRAINING_FORWARD_PROP_LAYERS",
        "TRAINING_BACK_PROP_ERRORS",
        "TRAINING_BACK_PROP_OUTPUT_LAYER",
        "TRAINING_BACK_PROP_HIDDEN_LAYERS",
        "TRAINING_UPDATE_WEIGHTS_WEIGHTS",
        "TRAINING_UPDATE_WEIGHTS_BIASES"
    ]

    # JOB IDS
    job_ids = []

    # GENERATE COMBINATIONS
    for r in range(len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            FLAGS_RAW = ",".join(combo)

            # RUN SCHEDULER
            res = subprocess.run(["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            job_id = re.search(r"Submitted batch job (\d+)", res.stdout.strip()).group(1)
            job_ids.append(job_id)

            if DEBUG: print(f"Submitted job {job_id}.")

    # WAIT FOR JOBS TO FINISH
    wait_time = 0
    while job_ids:
        # CHECK JOB STATUS
        res = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i", "-u", os.environ["USER"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # GET RUNNING JOBS AND UPDATE PENDING JOBs
        running_jobs = set(res.stdout.strip().split("\n")) if res.stdout.strip() else set()
        job_ids = [jid for jid in job_ids if jid in running_jobs]

        # WAIT BEFORE NEXT CHECK
        time.sleep(SLEEP_INTERVAL)
        wait_time += SLEEP_INTERVAL
        if DEBUG: print(f"{wait_time:3} s\t{len(job_ids):3} jobs remaining")

# ANALYSIS
if "a" in parts:
    # AGGREGATE RUNTIMES
    runtimes = defaultdict(list)

    # CRAWL OUTPUT FOLDER
    for filename in os.listdir(OUTPUT_DIR):
        # SKIP NON-OUTPUT FILES
        if not filename.endswith(".out"):
            continue

        # READ OUTPUT FILE
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "r") as f:
            content = f.read()
        
        # EXTRACT METADATA
        meta_match = re.search(r"---\s*(.*?)\s*---", content, re.DOTALL)
        if not meta_match:
            continue
        metadata = yaml.safe_load(meta_match.group(1))
        flags = metadata.get("compilation_flags", "")
        if flags: flags = flags.replace("-D", "")

        # EXTRACT RUNTIMES
        runtimes_list = [float(m.split()[1]) for m in re.findall(r"#START:\d+\s*(\d+\s+\d+\.\d+)", content)]
        if runtimes_list:
            #avg_runtime = sum(runtimes_list) / len(runtimes_list)
            runtimes[flags].extend(runtimes_list)

    # AVERAGE RUNTIMES
    # COMPUTE STATS
    stats = {}
    for flags, vals in runtimes.items():
        avg = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mn = min(vals)
        mx = max(vals)
        n = len(vals)
        std_var = std / avg
        min_max = mx / mn
        stats[flags] = (avg, std, mn, mx, n, std_var, min_max)

    # SORT
    ranking = sorted(stats.items(), key=lambda x: x[1])

    # RANKING
    with open(RESULTS_FILE, "a") as f:
        ind = 1
        f.write(f"\n# {SUBFOLDER}\n```\n")
        for flags, (avg, std, mn, mx, n, std_var, min_max) in ranking:
            f.write(f"{ind:03}. avg={avg:09.6f} | std={std:09.6f} | min={mn:09.6f} | max={mx:09.6f} | n={n:03} | rel_std={std_var:09.6f} | range_ratio={min_max:09.6f} | {flags}\n")
            ind += 1
        f.write("```\n")

