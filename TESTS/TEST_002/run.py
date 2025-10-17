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
SLEEP_INTERVAL = 1  # seconds
DEBUG          = True

# SCHEDULER ARGUMENTS
TEST_NUM    = "002"
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
TRAINING_FILE    = f"-Itraining {ROOT_DIR}/training.c"

# WHAT TO DO: execute (e), analyze (a)
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

# EXECUTION
if "e" in parts:
    # JOB IDS
    job_ids = []

    # SUBMIT ALL TAGS JOB
    res_tags = subprocess.run(["sbatch", SCHEDULER_FILE, TEST_NUM, "ALL", REPEAT, SUBFOLDER, TRAINING_FILE], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    tags_id = re.search(r"Submitted batch job (\d+)", res_tags.stdout.strip()).group(1)
    job_ids.append(tags_id)
    if DEBUG: print(f"Submitted job {tags_id}.")

    # SUBMIT NO TAGS JOB
    res_none = subprocess.run(["sbatch", SCHEDULER_FILE, TEST_NUM, "NONE", REPEAT, SUBFOLDER, TRAINING_FILE], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    none_id = re.search(r"Submitted batch job (\d+)", res_none.stdout.strip()).group(1)
    job_ids.append(none_id)
    if DEBUG: print(f"Submitted job {none_id}.")

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
    runtimes_par = defaultdict(list)
    runtimes_seq = defaultdict(list)
    tags = ["FI", "FPL", "BPE", "BPO", "BPH", "UWW", "UWB"]

    # CRAWL OUTPUT FOLDER
    for filename in os.listdir(OUTPUT_DIR):
        # SKIP NON-OUTPUT FILES
        if not filename.endswith(".out"):
            continue

        # READ OUTPUT FILE
        filepath = os.path.join(OUTPUT_DIR, filename)
        meta_lines = []
        in_meta = False
        tag_times = defaultdict(list)
        with open(filepath) as f:
            # ITERATE LINES
            for line in f:
                line = line.strip()

                # METADATA BLOCK
                if line.startswith("---"):
                    in_meta = not in_meta
                    continue
                if in_meta:
                    meta_lines.append(line)
                    continue

                # RUNTIMES
                if ":" in line:
                    tag, val = line.split(":", 1)
                    tag = tag.strip()
                    if tag in tags: tag_times[tag].append(float(val.strip()))

        # Parse metadata
        if meta_lines:
            metadata = yaml.safe_load("\n".join(meta_lines))
            flags = metadata.get("compilation_flags", "")
            if flags: flags = flags.replace("-D", "")
        
        # ASSIGN RUNTIMES
        if flags == "ALL":
            for t, vals in tag_times.items():
                runtimes_par[t].extend(vals)
        elif flags == "NONE":
            for t, vals in tag_times.items():
                runtimes_seq[t].extend(vals)

    # COMPUTE STATS
    tot_par = sum(sum(v) for v in runtimes_par.values())
    tot_seq = sum(sum(v) for v in runtimes_seq.values())
    stats = {}
    for tag in tags:
        time_par = sum(runtimes_par[tag])
        time_seq = sum(runtimes_seq[tag])
        fraction_par = time_par / tot_par if tot_par else 0
        fraction_seq = time_seq / tot_seq if tot_seq else 0
        speedup = time_seq / time_par if time_par else 0
        stats[tag] = (time_par, fraction_par, time_seq, fraction_seq, speedup)
    
    # SORT
    ranking = sorted(stats.items(), key=lambda x: x[1][4], reverse=True)

    # RANKING
    with open(RESULTS_FILE, "a") as f:
        ind = 1
        f.write(f"\n# {SUBFOLDER}\n```\n")
        for tag, (time_par, fraction_par, time_seq, fraction_seq, speedup) in ranking:
            f.write(f"{ind:03}. time_par={time_par:09.6f} | fraction_par={fraction_par:06.4f} | time_seq={time_seq:09.6f} | fraction_seq={fraction_seq:06.4f} | speedup={speedup:07.4f} | {tag}\n")
            ind += 1
        f.write("```\n")

