import argparse
import subprocess
import re
from time import sleep
import os

# Constants
SCHEDULER_FILE = "mpi.sub"

# Arguemnts
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="slurm")
parser.add_argument("--nprocs", type=int, default=8)
parser.add_argument("--partition", type=str, default="nodo.q")
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--num_neurons", type=int, default=135)
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

# Change scheduler file
with open(SCHEDULER_FILE, "r") as f:
    contents = f.read()
contents = re.sub(r"#SBATCH\s+-o\s+\S+", f"#SBATCH -o OUT/{args.name}-%j.out", contents)
contents = re.sub(r"#SBATCH\s+-e\s+\S+", f"#SBATCH -e OUT/{args.name}-%j.err", contents)
contents = re.sub(r"#SBATCH\s+--ntasks=\d+", f"#SBATCH --ntasks={args.nprocs}", contents)
contents = re.sub(r"#SBATCH\s+--partition=\S+", f"#SBATCH --partition={args.partition}", contents)
with open(SCHEDULER_FILE, "w") as f:
    f.write(contents)

# Submit job
res = subprocess.run(
    ["sbatch", SCHEDULER_FILE, str(args.num_epochs), str(args.num_neurons), str(args.repeat)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True
)
print(res.stdout.strip())
job_id = re.search(r"Submitted batch job (\d+)", res.stdout.strip()).group(1)

# Wait
while True:
    res = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%i", "-u", os.environ.get("USER", "capmc-1")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if str(job_id) not in res.stdout:
        break
    sleep(1)

# Open output file
subprocess.run(["code", f"OUT/{args.name}-{job_id}.out"])

