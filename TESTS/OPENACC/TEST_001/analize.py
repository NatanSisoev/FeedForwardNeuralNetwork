import os
import re
import csv

# -----------------------------
# CONFIGURACIÓ
# -----------------------------
INPUT_DIR = "./OUT/A"
OUTPUT_FILE = "resultats_A.csv"

START_ID = 107220
END_ID = 107235

# -----------------------------
# REGEX
# -----------------------------
re_test_number = re.compile(r"test_number:\s*(\d+)")
re_job_id = re.compile(r"job_id:\s*(\d+)")
re_flags = re.compile(r"compilation_flags:\s*(.*)")
re_gpu = re.compile(r"gpu_name:\s*(.*)")
re_start_line = re.compile(r"#START:\d+")
re_accuracy_time = re.compile(r"(\d+)\s+([\d\.]+)\s*sec")

rows = []

# -----------------------------
# LLEGIR FITXERS
# -----------------------------
for job_id in range(START_ID, END_ID + 1):
    filename = os.path.join(INPUT_DIR, "out_{}.out".format(job_id))
    print(f"[DEBUG] Processant fitxer: {filename}")

    if not os.path.exists(filename):
        print(f"   ❌ Fitxer no trobat")
        continue
    else:
        print(f"   ✔ Fitxer trobat")

    data = {
        "job_id": job_id,
        "test_number": None,
        "flags": "",
        "gpu_name": "",
        "accuracy": None,
        "time_sec": None
    }

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        m = re_test_number.search(line)
        if m:
            data["test_number"] = int(m.group(1))
            print(f"      [DEBUG] test_number = {data['test_number']}")
            continue

        m = re_flags.search(line)
        if m:
            data["flags"] = m.group(1).strip()
            print(f"      [DEBUG] flags = {data['flags']}")
            continue

        m = re_gpu.search(line)
        if m:
            data["gpu_name"] = m.group(1).strip()
            print(f"      [DEBUG] gpu_name = {data['gpu_name']}")
            continue

        if re_start_line.search(line):
            continue

        m = re_accuracy_time.search(line)
        if m:
            data["accuracy"] = int(m.group(1))
            data["time_sec"] = float(m.group(2))
            print(f"      [DEBUG] accuracy = {data['accuracy']}, time_sec = {data['time_sec']}")
            continue

    rows.append(data)
    print(f"   ✔ Dades llegides per job_id {job_id}")

# -----------------------------
# ESCRIURE RESULTATS
# -----------------------------
if rows:
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Resultats guardats a: {os.path.abspath(OUTPUT_FILE)}")
else:
    print("❌ No hi ha dades per escriure. Revisa la carpeta d'entrada i els noms dels fitxers.")
