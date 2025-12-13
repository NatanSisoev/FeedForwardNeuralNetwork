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
SLEEP_INTERVAL = 2
DEBUG = True

# SCHEDULER ARGUMENTS
TEST_NUM = "002"
ROOT_DIR = f"/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/TESTS/OPENMPI/TEST_{TEST_NUM}"
FLAGS_RAW = "ALL"
REPEAT = "1"
SUBFOLDER = sys.argv[1] if len(sys.argv) > 1 else "X"

# CONFIGURATIONS
BASE_DIR = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode"
CONFIG_TEMPLATE = f"{BASE_DIR}/configuration/configfile.txt"
CONFIG_DIR = f"{ROOT_DIR}/configs"
SERVER = "Wilma"

# EXPERIMENT CONFIGURATIONS
NUM_EPOCHS = [1, 10, 50, 100, 200]
NUM_TASKS = [2, 4, 6, 8, 12]
NUM_NODES = [1, 4, 8, 12]
NUM_NEURONS = [135, 250]

# CREATE DIRECTORIES
os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# CREATE NEW SUBFOLDER
if SUBFOLDER == "X":
    out_dir = f"{ROOT_DIR}/OUT"
    os.makedirs(out_dir, exist_ok=True)
    used = [d for d in os.listdir(out_dir) if d in string.ascii_uppercase and os.path.isdir(os.path.join(out_dir, d))]
    next_letter = chr(ord(max(used)) + 1) if used else "A"
    SUBFOLDER = next_letter
    os.makedirs(os.path.join(out_dir, next_letter))

# FILE PATHS
SCHEDULER_FILE = f"{BASE_DIR}/mpi.sub"
OUTPUT_DIR = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE = f"{ROOT_DIR}/results.md"
TRAINING_FILE = f"{BASE_DIR}/training/training.c"

# WHAT TO DO
parts = sys.argv[2] if len(sys.argv) > 2 else "ea"

def create_config_file(filename, num_epochs, num_neurons):
    """Crea un fitxer de configuració amb els paràmetres especificats"""
    config_content = f"""num_layers=3
layer=1024
layer={num_neurons}
layer=10
num_training_patterns=1934
num_test_patterns=946
img_dim_x=32
img_dim_y=32
dataset_training_path=./datasets/optdigits.tra
dataset_test_path=./datasets/optdigits.cv
num_epochs={num_epochs}
seed=50
alpha=0.15
debug=0
"""
    with open(filename, "w") as f:
        f.write(config_content)
    
    if DEBUG:
        print(f"  Created config: {os.path.basename(filename)}")

def modify_scheduler(num_tasks, num_nodes):
    """Modifica el mpi.sub temporalment per ajustar ntasks i nodes"""
    with open(SCHEDULER_FILE, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith("#SBATCH --ntasks="):
            new_lines.append(f"#SBATCH --ntasks={num_tasks}\n")
        elif line.startswith("#SBATCH -N ") or line.startswith("#SBATCH --nodes="):
            new_lines.append(f"#SBATCH -N {num_nodes}\n")
        else:
            new_lines.append(line)
    
    with open(SCHEDULER_FILE, "w") as f:
        f.writelines(new_lines)

def restore_scheduler():
    """Restaura el mpi.sub original (opcional)"""
    pass

# EXECUTION
if "e" in parts:
    print(f"Saving output to '{OUTPUT_DIR}'")
    print(f"Config files in '{CONFIG_DIR}'")

    job_ids = []
    config_index = 1

    # EXPERIMENT 2.1: Strong Scaling (processos)
    print("\n=== EXPERIMENT 2.1: Strong Scaling (Processos) ===")
    baseline_epochs = 10
    baseline_neurons = 135
    baseline_nodes = 1
    
    for num_tasks in NUM_TASKS:
        config_file = f"{CONFIG_DIR}/config_{config_index:03d}_tasks{num_tasks}_epochs{baseline_epochs}_neurons{baseline_neurons}_nodes{baseline_nodes}.txt"
        create_config_file(config_file, baseline_epochs, baseline_neurons)
        modify_scheduler(num_tasks, baseline_nodes)
        
        if DEBUG:
            print(f"Job {config_index}: tasks={num_tasks} epochs={baseline_epochs} neurons={baseline_neurons} nodes={baseline_nodes}")
        
        res = subprocess.run(
            ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        job_match = re.search(r"Submitted batch job (\d+)", res.stdout.strip())
        if job_match:
            job_id = job_match.group(1)
            job_ids.append(job_id)
            if DEBUG: print(f"  → Job ID: {job_id}")
        else:
            print(f"ERROR submitting job!")
            print(f"  STDOUT: {res.stdout}")
            print(f"  STDERR: {res.stderr}")
        
        config_index += 1
        time.sleep(0.5)  # Petit delay entre submissions

    # EXPERIMENT 2.2: Escalabilitat per èpoques
    print("\n=== EXPERIMENT 2.2: Escalabilitat per Èpoques ===")
    baseline_tasks = 8
    
    for num_epochs in NUM_EPOCHS:
        config_file = f"{CONFIG_DIR}/config_{config_index:03d}_tasks{baseline_tasks}_epochs{num_epochs}_neurons{baseline_neurons}_nodes{baseline_nodes}.txt"
        create_config_file(config_file, num_epochs, baseline_neurons)
        modify_scheduler(baseline_tasks, baseline_nodes)
        
        if DEBUG:
            print(f"Job {config_index}: epochs={num_epochs} tasks={baseline_tasks} neurons={baseline_neurons} nodes={baseline_nodes}")
        
        res = subprocess.run(
            ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        job_match = re.search(r"Submitted batch job (\d+)", res.stdout.strip())
        if job_match:
            job_id = job_match.group(1)
            job_ids.append(job_id)
            if DEBUG: print(f"  → Job ID: {job_id}")
        
        config_index += 1
        time.sleep(0.5)

    # EXPERIMENT 2.3: Escalabilitat per neurones
    print("\n=== EXPERIMENT 2.3: Escalabilitat per Neurones ===")
    
    for num_neurons in NUM_NEURONS:
        config_file = f"{CONFIG_DIR}/config_{config_index:03d}_tasks{baseline_tasks}_epochs{baseline_epochs}_neurons{num_neurons}_nodes{baseline_nodes}.txt"
        create_config_file(config_file, baseline_epochs, num_neurons)
        modify_scheduler(baseline_tasks, baseline_nodes)
        
        if DEBUG:
            print(f"Job {config_index}: neurons={num_neurons} tasks={baseline_tasks} epochs={baseline_epochs} nodes={baseline_nodes}")
        
        res = subprocess.run(
            ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        job_match = re.search(r"Submitted batch job (\d+)", res.stdout.strip())
        if job_match:
            job_id = job_match.group(1)
            job_ids.append(job_id)
            if DEBUG: print(f"  → Job ID: {job_id}")
        
        config_index += 1
        time.sleep(0.5)

    # EXPERIMENT 2.4: Weak Scaling (nodes)
    print("\n=== EXPERIMENT 2.4: Weak Scaling (Nodes) ===")
    tasks_per_node = 8
    
    for num_nodes in NUM_NODES:
        total_tasks = tasks_per_node * num_nodes
        config_file = f"{CONFIG_DIR}/config_{config_index:03d}_tasks{total_tasks}_epochs{baseline_epochs}_neurons{baseline_neurons}_nodes{num_nodes}.txt"
        create_config_file(config_file, baseline_epochs, baseline_neurons)
        modify_scheduler(total_tasks, num_nodes)
        
        if DEBUG:
            print(f"Job {config_index}: nodes={num_nodes} total_tasks={total_tasks} epochs={baseline_epochs} neurons={baseline_neurons}")
        
        res = subprocess.run(
            ["sbatch", SCHEDULER_FILE, TEST_NUM, FLAGS_RAW, REPEAT, SUBFOLDER, TRAINING_FILE, config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        job_match = re.search(r"Submitted batch job (\d+)", res.stdout.strip())
        if job_match:
            job_id = job_match.group(1)
            job_ids.append(job_id)
            if DEBUG: print(f"  → Job ID: {job_id}")
        
        config_index += 1
        time.sleep(0.5)

    print(f"\nTotal jobs submitted: {len(job_ids)}")

    # WAIT FOR JOBS
    if job_ids:
        print("\nWaiting for jobs to complete...")
        wait_time = 0
        num_jobs = len(job_ids)
        while job_ids:
            res = subprocess.run(
                ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i", "-u", os.environ.get("USER", "capmc-1")],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            running_jobs = set(res.stdout.strip().split("\n")) if res.stdout.strip() else set()
            job_ids = [jid for jid in job_ids if jid in running_jobs]

            if job_ids:
                time.sleep(SLEEP_INTERVAL)
                wait_time += SLEEP_INTERVAL
                if DEBUG: 
                    print(f"{wait_time:3}s\t{len(job_ids):2}/{num_jobs:2} jobs remaining", end="\r")
        
        print(f"\n✓ All jobs completed in {wait_time}s")

# ANALYSIS
if "a" in parts:
    print(f"\nAnalyzing results from '{OUTPUT_DIR}'...")
    
    # DATASET: epochs -> tasks -> nodes -> neurons -> list of (train_time, test_time, total_time, accuracy)
    dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    files_processed = 0
    for filename in os.listdir(OUTPUT_DIR):
        if not filename.endswith(".out") or filename.startswith("slurm-") or filename.startswith("err_"):
            continue
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        meta_lines = []
        in_meta = False
        train_time = test_time = total_time = accuracy = None

        with open(filepath) as f:
            for line in f:
                line_stripped = line.strip()
                
                # Parse metadata
                if line_stripped.startswith("---"):
                    in_meta = not in_meta
                    continue
                if in_meta:
                    meta_lines.append(line_stripped)
                    continue
                
                # Parse timing (busquem línies com: TRAIN_TIME: 1.234)
                if "TRAIN_TIME" in line_stripped:
                    try:
                        train_time = float(line_stripped.split()[-1])
                    except:
                        pass
                elif "TEST_TIME" in line_stripped:
                    try:
                        test_time = float(line_stripped.split()[-1])
                    except:
                        pass
                elif "TOTAL_TIME" in line_stripped:
                    try:
                        total_time = float(line_stripped.split()[-1])
                    except:
                        pass
                elif "ccuracy" in line_stripped.lower():  # Accuracy o accuracy
                    try:
                        nums = re.findall(r'\d+', line_stripped)
                        if nums:
                            accuracy = int(nums[-1])
                    except:
                        pass

        if not meta_lines:
            continue
        
        try:
            metadata = yaml.safe_load("\n".join(meta_lines))
            num_epochs = int(metadata.get("num_epochs", 0))
            num_tasks = int(metadata.get("num_processes", 0))
            num_neurons = int(metadata.get("num_neurons", 0))
            
            # Extreure num_nodes del nom del config file si està disponible
            num_nodes = 1  # default
            if "output_file" in metadata:
                # Intentar parsejar dels logs o assumir 1 node per defecte
                pass

            if train_time and test_time:
                if total_time is None:
                    total_time = train_time + test_time
                dataset[num_epochs][num_tasks][num_nodes][num_neurons].append(
                    (train_time, test_time, total_time, accuracy)
                )
                files_processed += 1
        except Exception as e:
            if DEBUG:
                print(f"Warning: Could not parse {filename}: {e}")
            continue

    print(f"Processed {files_processed} output files")

    # WRITE RESULTS
    with open(RESULTS_FILE, "w") as f:
        f.write(f"# TEST_002 Results - Subfolder {SUBFOLDER}\n")
        f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # EXPERIMENT 2.1
        f.write("## Experiment 2.1: Strong Scaling (Processos)\n\n")
        f.write("| Tasks | Train (s) | Test (s) | Total (s) | Speedup | Efficiency (%) |\n")
        f.write("|-------|-----------|----------|-----------|---------|----------------|\n")
        
        baseline_time = None
        for tasks in sorted(NUM_TASKS):
            data = dataset[10][tasks][1][135]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                
                if baseline_time is None:
                    baseline_time = total_avg
                
                speedup = baseline_time / total_avg if total_avg > 0 else 0
                efficiency = (speedup / tasks) * 100 if tasks > 0 else 0
                f.write(f"| {tasks} | {train_avg:.4f} | {test_avg:.4f} | {total_avg:.4f} | {speedup:.2f} | {efficiency:.1f} |\n")

        # EXPERIMENT 2.2
        f.write("\n## Experiment 2.2: Escalabilitat per Èpoques\n\n")
        f.write("| Epochs | Train (s) | Test (s) | Total (s) | Accuracy |\n")
        f.write("|--------|-----------|----------|-----------|----------|\n")
        
        for epochs in sorted(NUM_EPOCHS):
            data = dataset[epochs][8][1][135]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                acc_list = [d[3] for d in data if d[3] is not None]
                acc_avg = sum(acc_list) / len(acc_list) if acc_list else 0
                f.write(f"| {epochs} | {train_avg:.4f} | {test_avg:.4f} | {total_avg:.4f} | {acc_avg:.0f} |\n")

        # EXPERIMENT 2.3
        f.write("\n## Experiment 2.3: Escalabilitat per Neurones\n\n")
        f.write("| Neurons | Train (s) | Test (s) | Total (s) | Accuracy |\n")
        f.write("|---------|-----------|----------|-----------|----------|\n")
        
        for neurons in sorted(NUM_NEURONS):
            data = dataset[10][8][1][neurons]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                acc_list = [d[3] for d in data if d[3] is not None]
                acc_avg = sum(acc_list) / len(acc_list) if acc_list else 0
                f.write(f"| {neurons} | {train_avg:.4f} | {test_avg:.4f} | {total_avg:.4f} | {acc_avg:.0f} |\n")

        # EXPERIMENT 2.4
        f.write("\n## Experiment 2.4: Weak Scaling (Nodes)\n\n")
        f.write("| Nodes | Total Tasks | Train (s) | Test (s) | Total (s) | Speedup |\n")
        f.write("|-------|-------------|-----------|----------|-----------|----------|\n")
        
        baseline_node_time = None
        for num_nodes in sorted(NUM_NODES):
            total_tasks = num_nodes * 8
            data = dataset[10][total_tasks][num_nodes][135]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                
                if baseline_node_time is None:
                    baseline_node_time = total_avg
                
                speedup = baseline_node_time / total_avg if total_avg > 0 else 0
                f.write(f"| {num_nodes} | {total_tasks} | {train_avg:.4f} | {test_avg:.4f} | {total_avg:.4f} | {speedup:.2f} |\n")

        f.write("\n---\n")

    print(f"✓ Results written to '{RESULTS_FILE}'")

print("\n✓ Done!")
