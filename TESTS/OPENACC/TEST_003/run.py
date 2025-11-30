import os
import sys
import subprocess
import re
import string
import time
import yaml
import shutil
from collections import defaultdict

# SETTINGS
SLEEP_INTERVAL = 1  # seconds
DEBUG          = True

# SCHEDULER ARGUMENTS
TEST_NUM    = "003"
ROOT_DIR    = f"/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/TESTS/OPENACC/TEST_{TEST_NUM}"
FLAGS_RAW   = "ALL"
REPEAT      = "1"
SUBFOLDER   = sys.argv[1] if len(sys.argv) > 1 else "X"

# CONFIGURATIONS
CONFIG_FILE = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/configuration/configfile.txt"
CONFIG_DIR  = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/configuration/"
SERVER      = "Aolin"
PARTITION   = "cuda-ext.q"  # GPU partition for external students

# GPU CONFIGURATIONS
# Basant-nos en la sortida de sinfo, les GPUs disponibles són:
# aolin-gpu-1: gpu:GeForceRTX2070:1 i gpu:GeForceGTX1080Ti:1
# aolin-gpu-2: gpu:GeForceGTX1080Ti:1 i gpu:GeForceGTX1080:1
# aolin-gpu-3 i aolin-gpu-4: gpu:GeForceRTX3080:1
GPU_TYPES = [
    "gpu:GeForceRTX3080:1",
    "gpu:GeForceRTX2070:1",
    "gpu:GeForceGTX1080Ti:1",
    "gpu:GeForceGTX1080:1"  # Afegida també la GTX1080
]

# NEURON CONFIGURATIONS
NUM_NEURONS = [135, 250, 2048]

# MODE (can be extended if needed)
mode = sys.argv[3] if len(sys.argv) > 3 else ""
if "N" in mode:
    NUM_NEURONS = [135, 250, 2048]
if "G" in mode:
    # All GPUs (already set)
    pass

# CREATE NEW SUBFOLDER (IF NONE INDICATED)
if SUBFOLDER == "X":
    os.makedirs(ROOT_DIR, exist_ok=True)
    out_dir = f"{ROOT_DIR}/OUT"
    os.makedirs(out_dir, exist_ok=True)
    used = [d for d in os.listdir(out_dir) if d in string.ascii_uppercase and os.path.isdir(os.path.join(out_dir, d))]
    next_letter = chr(ord(max(used)) + 1) if used else "A"
    SUBFOLDER = next_letter
    os.makedirs(os.path.join(out_dir, next_letter))
os.makedirs(CONFIG_DIR, exist_ok=True)

# FILE PATHS
SCHEDULER_FILE   = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/scheduler.sub"
OUTPUT_DIR       = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE     = f"{ROOT_DIR}/results.md"
TRAINING_FILE    = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/training/training.c"

# WHAT TO DO: execute (e), analyze (a)
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

# CONSTANTS
SCHEDULER_LINES = []

# CHANGE CONFIG AND SCHEDULER FILE FOR GPU
def change_config_scheduler_gpu(job_config, partition, gpu_type, num_neurons):
    global SCHEDULER_LINES

    # READ AND SAVE CONFIG FILE
    with open(job_config, "r") as f:
        config_lines = f.readlines()

    # MODIFY CONFIGURATION - neurons (line 3, index 2)
    config_lines[2] = f"layer={num_neurons}\n"

    # WRITE TEMP CONFIG FILE
    with open(job_config, "w") as f:
        f.writelines(config_lines)

    # READ AND SAVE SCHEDULER
    if SCHEDULER_LINES == []:
        with open(SCHEDULER_FILE, "r") as f:
            SCHEDULER_LINES = f.readlines()

    # MODIFY SCHEDULER FOR GPU
    new_scheduler_lines = SCHEDULER_LINES.copy()
    
    # Modify partition and gres lines
    for i, line in enumerate(new_scheduler_lines):
        if line.startswith("#SBATCH --partition="):
            new_scheduler_lines[i] = f"#SBATCH --partition={partition}\n"
        elif line.startswith("#SBATCH --gres="):
            new_scheduler_lines[i] = f"#SBATCH --gres={gpu_type}\n"

    # WRITE TEMP SCHEDULER FILE
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

    for gpu_type in GPU_TYPES:
        for num_neurons in NUM_NEURONS:
            # JOB CONFIG FILE
            job_config = f"{CONFIG_DIR}/configfile_{config_file_index}.txt"
            shutil.copy(CONFIG_FILE, job_config)

            # CHANGE CONFIG AND SCHEDULER FILES
            change_config_scheduler_gpu(job_config, PARTITION, gpu_type, num_neurons)

            # PRINT INFO
            gpu_name = gpu_type.split(":")[1]  # Extract GPU name
            if DEBUG: 
                print(f"Submitting job {config_file_index}: GPU={gpu_name} | neurons={num_neurons}")

            # SUBMIT JOB
            res = subprocess.run(
                ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, job_config],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Check if job was submitted successfully
            if DEBUG:
                print(f"  STDOUT: {res.stdout.strip()}")
                if res.stderr:
                    print(f"  STDERR: {res.stderr.strip()}")
            
            job_match = re.search(r"Submitted batch job (\d+)", res.stdout.strip())
            if job_match:
                job_id = job_match.group(1)
                job_ids.append(job_id)
                config_file_index += 1
                if DEBUG: print(f"  → Job ID: {job_id}")
            else:
                print(f"ERROR submitting job!")
                print(f"  Command output: {res.stdout}")
                print(f"  Command errors: {res.stderr}")

    print(f"\nTotal jobs submitted: {len(job_ids)}")

    # WAIT FOR JOBS TO FINISH
    if job_ids:
        print("\nWaiting for jobs to complete...")
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

            # GET RUNNING JOBS AND UPDATE PENDING JOBS
            running_jobs = set(res.stdout.strip().split("\n")) if res.stdout.strip() else set()
            job_ids = [jid for jid in job_ids if jid in running_jobs]

            # WAIT BEFORE NEXT CHECK
            if job_ids:  # Only sleep if there are still jobs running
                time.sleep(SLEEP_INTERVAL)
                wait_time += SLEEP_INTERVAL
                if DEBUG: print(f"{wait_time:3}s\t{len(job_ids):2}/{num_jobs:2} jobs remaining", end="\r")
        
        print(f"\n\n✓ All jobs completed in {wait_time}s")

# ANALYSIS
if "a" in parts:
    print(f"\nAnalyzing results from '{OUTPUT_DIR}'...")
    
    # DATASET: server -> partition -> gpu -> neurons -> list of (hits, runtime)
    dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # CRAWL OUTPUT FOLDER
    files_processed = 0
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".out") or filename.startswith("slurm-"):
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
                # Look for START/END markers and extract data between them
                if line.startswith("#START:"):
                    continue
                if line.startswith("#END:"):
                    continue
                # Parse hits and runtime (format: "hits\truntime")
                if "\t" in line:
                    try:
                        hits, runtime = line.split("\t")
                        hits_runtime.append((int(hits), float(runtime)))
                    except ValueError:
                        continue

        if not meta_lines: 
            continue
        
        metadata = yaml.safe_load("\n".join(meta_lines))
        server = metadata.get("server", "unknown")
        partition = metadata.get("partition_file", "unknown")
        gpu_name = metadata.get("gpu_name", "unknown")
        num_neurons = int(metadata.get("num_neurons", 0))

        if hits_runtime:
            dataset[server][partition][gpu_name][num_neurons].extend(hits_runtime)
            files_processed += 1

    print(f"Processed {files_processed} output files")

    # WRITE RESULTS
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n# GPU Test Results - Subfolder {SUBFOLDER}\n")
        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        for server, partitions in dataset.items():
            for partition, gpu_data in partitions.items():
                f.write(f"\n## Server: `{server}` | Partition: `{partition}`\n")
                
                all_gpus = sorted(gpu_data.keys())
                all_neurons = sorted({n for gpu in gpu_data.values() for n in gpu})

                if not all_gpus or not all_neurons:
                    f.write("*No data available*\n")
                    continue

                # --- Runtime table (GPU x Neurons) ---
                f.write(f"\n### ⏱️ Runtime (seconds)\n")
                header = "| GPU | " + " | ".join(f"{n} neurons" for n in all_neurons) + " |\n"
                f.write(header)
                f.write("|" + "---|"*(len(all_neurons)+1) + "\n")
                
                for gpu_name in all_gpus:
                    row = [f"**{gpu_name}**"]
                    for num_neurons in all_neurons:
                        vals = gpu_data[gpu_name].get(num_neurons, [])
                        time_avg = sum(d[1] for d in vals)/len(vals) if vals else 0
                        row.append(f"{time_avg:.6f}")
                    f.write("| " + " | ".join(row) + " |\n")

                # --- Hits table (GPU x Neurons) ---
                f.write(f"\n### 🎯 Hits (accuracy)\n")
                header = "| GPU | " + " | ".join(f"{n} neurons" for n in all_neurons) + " |\n"
                f.write(header)
                f.write("|" + "---|"*(len(all_neurons)+1) + "\n")
                
                for gpu_name in all_gpus:
                    row = [f"**{gpu_name}**"]
                    for num_neurons in all_neurons:
                        vals = gpu_data[gpu_name].get(num_neurons, [])
                        hits_avg = sum(d[0] for d in vals)/len(vals) if vals else 0
                        row.append(f"{hits_avg:.2f}")
                    f.write("| " + " | ".join(row) + " |\n")

                # --- Speedup relative to slowest GPU ---
                f.write(f"\n### 🚀 Speedup (relative to slowest GPU)\n")
                header = "| GPU | " + " | ".join(f"{n} neurons" for n in all_neurons) + " |\n"
                f.write(header)
                f.write("|" + "---|"*(len(all_neurons)+1) + "\n")
                
                # Find slowest time for each neuron config
                slowest_times = {}
                for num_neurons in all_neurons:
                    max_time = 0
                    for gpu_name in all_gpus:
                        vals = gpu_data[gpu_name].get(num_neurons, [])
                        avg_time = sum(d[1] for d in vals)/len(vals) if vals else 0
                        max_time = max(max_time, avg_time)
                    slowest_times[num_neurons] = max_time
                
                for gpu_name in all_gpus:
                    row = [f"**{gpu_name}**"]
                    for num_neurons in all_neurons:
                        vals = gpu_data[gpu_name].get(num_neurons, [])
                        avg_time = sum(d[1] for d in vals)/len(vals) if vals else 0
                        speedup = slowest_times[num_neurons]/avg_time if avg_time else 0
                        row.append(f"{speedup:.2f}x")
                    f.write("| " + " | ".join(row) + " |\n")

                f.write("\n---\n")

    print(f"✓ Results written to '{RESULTS_FILE}'")

print("\n✓ Done!")