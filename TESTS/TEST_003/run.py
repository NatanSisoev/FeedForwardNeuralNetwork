import os
import sys
import subprocess
import itertools
import re
import string
import time
import yaml
import statistics
import shutil
from collections import defaultdict

# SETTINGS
SLEEP_INTERVAL = 1  # seconds
DEBUG          = True

# SCHEDULER ARGUMENTS
TEST_NUM    = "003"
ROOT_DIR    = f"TESTS/TEST_{TEST_NUM}"
FLAGS_RAW   = "OPT"
REPEAT      = "1"
SUBFOLDER   = sys.argv[1] if len(sys.argv) > 1 else "X"

# CONFIGURATIONS
CONFIG_FILE = "configuration/configfile.txt"
CONFIG_DIR  = f"{ROOT_DIR}/configuration"
SERVER      = "Wilma"         # Wilma | Aolin
PARTITION   = ["nodo.q"]      # nodo.q | new-nodo.q | aolin.q
NUM_THREADS = [12]            # 1 | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 32 | 64
NUM_EPOCHS  = [10]            # 1 | 10 | 100 | 1000
NUM_NEURONS = [135]           # 135 | 250

# MODE
mode = sys.argv[3] if len(sys.argv) > 3 else ""
if "P" in mode:
    if SERVER == "Wilma":
        PARTITION = ["nodo.q", "new-nodo.q"]
    elif SERVER == "Aolin":
        PARTITION = ["aolin.q"]
if "T" in mode:
    NUM_THREADS = [1, 2, 4, 6, 8, 10, 12, 32, 64]
if "E" in mode:
    NUM_EPOCHS = [1, 10, 100, 1000]
if "N" in mode:
    NUM_NEURONS = [135, 250]

# CREATE NEW SUBFOLDER (IF NONE INDICATED)
if SUBFOLDER == "X":
    os.makedirs(ROOT_DIR, exist_ok=True)
    used = [d for d in os.listdir(f"{ROOT_DIR}/OUT") if d in string.ascii_uppercase]
    next_letter = chr(ord(max(used)) + 1) if used else "A"
    SUBFOLDER = next_letter
    os.makedirs(os.path.join(f"{ROOT_DIR}/OUT", next_letter))
os.makedirs(CONFIG_DIR, exist_ok=True)

# FILE PATHS
SCHEDULER_FILE   = "scheduler.sub"
OUTPUT_DIR       = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE     = f"{ROOT_DIR}/results.md"
TRAINING_FILE    = f"training/training.c"

# WHAT TO DO: execute (e), analyze (a)
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

# CONSTANTS
SCHEDULER_LINES = []

# CHANGE CONFIG AND SCHEDULER FILE
def change_config_scheduler(job_config, partition, num_threads, num_epochs, num_neurons):
    global SCHEDULER_LINES

    # READ AND SAVE CONFIG FILE
    with open(job_config, "r") as f:
        config_lines = f.readlines()

    # MODIFY CONFIGURATION
    config_lines[2] = f"layer={num_neurons}\n"
    config_lines[10] = f"num_epochs={num_epochs}\n"

    # WRITE TEMP CONFIG FILE
    with open(job_config, "w") as f:
        f.writelines(config_lines)

    # READ AND SAVE SCHEDULER
    if SCHEDULER_LINES == []:
        with open(SCHEDULER_FILE, "r") as f:
            SCHEDULER_LINES = f.readlines()

    # MODIFY CONFIGURATION
    new_scheduler_lines = SCHEDULER_LINES.copy()
    new_scheduler_lines[3] = f"#SBATCH --partition={partition}\n"
    new_scheduler_lines[7] = f"export OMP_NUM_THREADS={num_threads}\n"

    # WRITE TEMP CONFIG FILE
    with open(SCHEDULER_FILE, "w") as f:
        f.writelines(new_scheduler_lines)

# EXECUTION
if "e" in parts:
    # FOLDER INFO
    print(f"Saving output to '{OUTPUT_DIR}' and results to '{RESULTS_FILE}'.")

    # JOB IDS
    job_ids = []
    pattern = re.compile(r"configfile_(\d+)\.txt$")
    m = lambda x : pattern.match(x)
    config_file_index = max((int(m(f).group(1)) for f in os.listdir(CONFIG_DIR) if m(f)), default=0) + 1

    for partition in PARTITION:
        if partition == "nodo.q":
            NUM_THREADS = [num for num in NUM_THREADS if num <= 12]
        elif partition == "new-nodo.q" or partition == "aolin.q":
            NUM_THREADS = [num for num in NUM_THREADS if num <= 64]
        for num_threads in NUM_THREADS:
            for num_epochs in NUM_EPOCHS:
                for num_neurons in NUM_NEURONS:
                    # JOB CONFIG FILE
                    job_config = f"{CONFIG_DIR}/configfile_{config_file_index}.txt"
                    shutil.copy(CONFIG_FILE, job_config)

                    # CHANGE CONFIG AND SCHEDULER FILES
                    change_config_scheduler(job_config, partition, num_threads, num_epochs, num_neurons)

                    # PRINT INFO
                    if DEBUG: print(f"Submitting job: server={SERVER} | partition={partition} | threads={num_threads} | epochs={num_epochs} | neurons={num_neurons}")

                    # SUBMIT JOB
                    res = subprocess.run(
                        ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, job_config],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    job_id = re.search(r"Submitted batch job (\d+)", res.stdout.strip()).group(1)
                    job_ids.append(job_id)
                    config_file_index += 1
                    if DEBUG: print(f"Submitted job {job_id}.")

    # WAIT FOR JOBS TO FINISH
    wait_time = 0
    num_jobs = len(job_ids)
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
        if DEBUG: print(f"{wait_time:3} s\t{len(job_ids):3} / {num_jobs:3} jobs remaining")

# ANALYSIS
if "a" in parts:
    # DATASET
    dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))

    # CRAWL OUTPUT FOLDER
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".out"):
            continue
        filepath = os.path.join(OUTPUT_DIR, filename)
        meta_lines = []
        in_meta = False
        hits_runtime = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith("---"):
                    in_meta = not in_meta
                    continue
                if in_meta:
                    meta_lines.append(line)
                    continue
                if "\t" in line:
                    hits, runtime = line.split("\t")
                    hits_runtime.append((int(hits), float(runtime)))

        if not meta_lines: continue
        metadata = yaml.safe_load("\n".join(meta_lines))
        server = metadata.get("server")
        partition = metadata.get("partition_file")
        num_threads = int(metadata.get("num_threads"))
        num_epochs = int(metadata.get("num_epochs"))
        num_neurons = int(metadata.get("num_neurons"))

        dataset[server][partition][num_threads][num_epochs][num_neurons].extend(hits_runtime)

    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n# {SUBFOLDER}\n")

        for server, partitions in dataset.items():
            for partition, threads_data in partitions.items():
                f.write(f"\n## Server: `{server}` - Partition: `{partition}`\n")
                all_neurons = sorted({n for nt in threads_data.values() for ep in nt.values() for n in ep})
                all_epochs = sorted({ep for nt in threads_data.values() for ep in nt})

                for neuron in all_neurons:
                    f.write(f"\n### Neurons: {neuron}\n")

                    # --- Runtime table ---
                    f.write(f"\n#### Runtime (threads x epochs)\n")
                    header = "| Threads | " + " | ".join(str(ep) for ep in all_epochs) + " |\n"
                    f.write(header)
                    f.write("|" + "---|"*(len(all_epochs)+1) + "\n")
                    for num_threads in sorted(threads_data):
                        row = [str(num_threads)]
                        for num_epochs in all_epochs:
                            vals = threads_data[num_threads].get(num_epochs, {}).get(neuron, [])
                            time_avg = sum(d[1] for d in vals)/len(vals) if vals else 0
                            row.append(f"{time_avg:.6f}")
                        f.write("| " + " | ".join(row) + " |\n")

                    # --- Speedup table ---
                    f.write(f"\n#### Speedup\n")
                    header = "| Threads | " + " | ".join(str(ep) for ep in all_epochs) + " |\n"
                    f.write(header)
                    f.write("|" + "---|"*(len(all_epochs)+1) + "\n")
                    base_times = {}
                    for num_epochs in all_epochs:
                        vals = threads_data[1].get(num_epochs, {}).get(neuron, [])
                        base_times[num_epochs] = sum(d[1] for d in vals)/len(vals) if vals else 0
                    for num_threads in sorted(threads_data):
                        row = [str(num_threads)]
                        for num_epochs in all_epochs:
                            vals = threads_data[num_threads].get(num_epochs, {}).get(neuron, [])
                            avg_time = sum(d[1] for d in vals)/len(vals) if vals else 0
                            speedup = base_times[num_epochs]/avg_time if avg_time else 0
                            row.append(f"{speedup:.4f}")
                        f.write("| " + " | ".join(row) + " |\n")

                    # --- Efficiency table ---
                    f.write(f"\n#### Efficiency (speedup / threads)\n")
                    header = "| Threads | " + " | ".join(str(ep) for ep in all_epochs) + " |\n"
                    f.write(header)
                    f.write("|" + "---|"*(len(all_epochs)+1) + "\n")
                    for num_threads in sorted(threads_data):
                        row = [str(num_threads)]
                        for num_epochs in all_epochs:
                            vals = threads_data[num_threads].get(num_epochs, {}).get(neuron, [])
                            avg_time = sum(d[1] for d in vals)/len(vals) if vals else 0
                            speedup = base_times[num_epochs]/avg_time if avg_time else 0
                            efficiency = speedup / num_threads if num_threads else 0
                            row.append(f"{efficiency:.4f}")
                        f.write("| " + " | ".join(row) + " |\n")

                # --- Hits table (neurons x epochs averaged over threads) ---
                f.write(f"\n### Hits Table\n")
                header = "| Neurons | " + " | ".join(str(ep) for ep in all_epochs) + " |\n"
                f.write(header)
                f.write("|" + "---|"*(len(all_epochs)+1) + "\n")
                for neuron in all_neurons:
                    row = [str(neuron)]
                    for num_epochs in all_epochs:
                        thread_vals = []
                        for num_threads in threads_data:
                            vals = threads_data[num_threads].get(num_epochs, {}).get(neuron, [])
                            thread_vals.extend(d[0] for d in vals)
                        hits_avg = sum(thread_vals)/len(thread_vals) if thread_vals else 0
                        row.append(f"{hits_avg:.4f}")
                    f.write("| " + " | ".join(row) + " |\n")
