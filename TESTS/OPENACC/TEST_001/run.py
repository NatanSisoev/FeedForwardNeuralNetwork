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
ROOT_DIR    = f"/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/TESTS/OPENACC/TEST_{TEST_NUM}"
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
SCHEDULER_FILE = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/scheduler.sub"
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
        "FORWARD_PROP",
        "BACK_PROP",
        "UPDATE_WEIGHTS"
    ]

    # JOB IDS
    job_ids = []

    # GENERATE COMBINATIONS
    for r in range(len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            FLAGS_RAW = ",".join(combo)

            # RUN SCHEDULER
            res = subprocess.run(["sbatch", SCHEDULER_FILE, TEST_NUM, \
                FLAGS_RAW, REPEAT, SUBFOLDER], stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE, universal_newlines=True, \
                cwd=os.path.dirname(SCHEDULER_FILE)
            )

            #print("STDOUT:", res.stdout)
            #print("STDERR:", res.stderr)
            #print("Return code:", res.returncode)

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

        # EXTRACT (accuracy, time)
        entries = re.findall(r"#START:\d+\s*(\d+)\s+(\d+\.\d+)", content)
        for acc, t in entries:
            stats[flags].append((int(acc), float(t)))

    # COMPUTE FINAL STATISTICS
    final = {}
    for flags, values in stats.items():

        accs  = [v[0] for v in values]
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

    # SORT BY AVERAGE TIME
    ranking = sorted(final.items(), key=lambda x: x[1][0])

    # WRITE RESULTS
    with open(RESULTS_FILE, "a") as f:
        ind = 1
        f.write(f"\n# {SUBFOLDER}\n```\n")

        for flags, (avg_t, std_t, min_t, max_t,
                    avg_a, std_a, min_a, max_a, n) in ranking:

            f.write(
                f"{ind:03}. "
                f"T(avg={avg_t:09.6f}, std={std_t:09.6f}, min={min_t:09.6f}, max={max_t:09.6f}) | "
                f"A(avg={avg_a:05.2f}, std={std_a:05.2f}, min={min_a:03d}, max={max_a:03d}) | "
                f"n={n:03} | {flags}\n"
            )
            ind += 1

        f.write("```\n")
