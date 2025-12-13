#!/usr/bin/env python3
"""
Script per analitzar els resultats del TEST_002
Executa: python3 analyze_results.py I
On 'I' és la subcarpeta dins de OUT/
"""

import os
import sys
import re
import yaml
from collections import defaultdict

# CONFIGURACIÓ
TEST_NUM = "002"
BASE_DIR = "/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode"
ROOT_DIR = f"{BASE_DIR}/TESTS/OPENMPI/TEST_{TEST_NUM}"

# Obtenir subfolder dels arguments
SUBFOLDER = sys.argv[1] if len(sys.argv) > 1 else "I"
OUTPUT_DIR = f"{ROOT_DIR}/OUT/{SUBFOLDER}"
RESULTS_FILE = f"{ROOT_DIR}/results.md"

print(f"Analyzing results from '{OUTPUT_DIR}'...")
print(f"Output will be written to '{RESULTS_FILE}'")

# Verificar que el directori existeix
if not os.path.exists(OUTPUT_DIR):
    print(f"ERROR: Directory {OUTPUT_DIR} does not exist!")
    sys.exit(1)

# ESTRUCTURA DE DADES
# dataset[num_epochs][num_tasks][num_nodes][num_neurons] = [(train_time, test_time, total_time, accuracy), ...]
dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

files_processed = 0
files_with_errors = []

# PROCESSAR TOTS ELS FITXERS .out
for filename in sorted(os.listdir(OUTPUT_DIR)):
    if not filename.startswith("out_") or not filename.endswith(".out"):
        continue
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Variables per parsejar
    meta_lines = []
    in_meta = False
    
    # Llistes per múltiples execucions
    train_times = []
    test_times = []
    total_times = []
    accuracies = []
    
    current_execution = {}
    
    # Llegir el fitxer
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                
                # Parsejar metadata (entre ---)
                if line_stripped == "---":
                    in_meta = not in_meta
                    continue
                
                if in_meta:
                    meta_lines.append(line_stripped)
                    continue
                
                # Detectar inici d'execució
                if line_stripped.startswith("#START:"):
                    current_execution = {}
                
                # Parsejar temps d'execució i accuracy
                # Format: "914\tTRAIN_TIME: 1.360084 sec"
                if "TRAIN_TIME" in line_stripped:
                    # Buscar accuracy (número abans de TRAIN_TIME)
                    acc_match = re.search(r'^(\d+)\s+TRAIN_TIME', line_stripped)
                    if acc_match:
                        current_execution['accuracy'] = int(acc_match.group(1))
                    
                    # Buscar temps
                    time_match = re.search(r'TRAIN_TIME[:\s]+([0-9.]+)', line_stripped)
                    if time_match:
                        current_execution['train_time'] = float(time_match.group(1))
                
                elif "TEST_TIME" in line_stripped:
                    time_match = re.search(r'TEST_TIME[:\s]+([0-9.]+)', line_stripped)
                    if time_match:
                        current_execution['test_time'] = float(time_match.group(1))
                
                elif "TOTAL_TIME" in line_stripped:
                    time_match = re.search(r'TOTAL_TIME[:\s]+([0-9.]+)', line_stripped)
                    if time_match:
                        current_execution['total_time'] = float(time_match.group(1))
                
                # Detectar final d'execució
                elif line_stripped.startswith("#END:"):
                    # Guardar dades d'aquesta execució
                    if 'train_time' in current_execution:
                        train_times.append(current_execution.get('train_time'))
                        test_times.append(current_execution.get('test_time'))
                        total_times.append(current_execution.get('total_time'))
                        if 'accuracy' in current_execution:
                            accuracies.append(current_execution.get('accuracy'))
                    current_execution = {}
        
        # Si no hem trobat metadata, saltar aquest fitxer
        if not meta_lines:
            continue
        
        # Parsejar metadata
        metadata = yaml.safe_load("\n".join(meta_lines))
        num_epochs = int(metadata.get("num_epochs", 0))
        num_tasks = int(metadata.get("num_processes", 0))
        num_nodes = int(metadata.get("num_nodes", 1))
        num_neurons = int(metadata.get("num_neurons", 0))
        num_executions = int(metadata.get("number_executions", 1))
        
        # Si falten dades crítiques, saltar
        if num_epochs == 0 or num_tasks == 0 or num_neurons == 0:
            files_with_errors.append((filename, "Missing critical metadata"))
            continue
        
        # Calcular mitjanes de totes les execucions
        if train_times and test_times and total_times:
            train_avg = sum(train_times) / len(train_times)
            test_avg = sum(test_times) / len(test_times)
            total_avg = sum(total_times) / len(total_times)
            acc_avg = sum(accuracies) / len(accuracies) if accuracies else None
            
            # Afegir a dataset
            dataset[num_epochs][num_tasks][num_nodes][num_neurons].append(
                (train_avg, test_avg, total_avg, acc_avg)
            )
            files_processed += 1
            
            num_runs = len(train_times)
            acc_str = f", accuracy={acc_avg:.1f}" if acc_avg else ""
            print(f"✓ {filename}: epochs={num_epochs}, tasks={num_tasks}, nodes={num_nodes}, neurons={num_neurons}, runs={num_runs}{acc_str}")
        else:
            files_with_errors.append((filename, f"Missing timing data (found {len(train_times)} executions)"))
    
    except Exception as e:
        files_with_errors.append((filename, str(e)))
        continue

print(f"\n{'='*60}")
print(f"Processed {files_processed} files successfully")
if files_with_errors:
    print(f"Skipped {len(files_with_errors)} files with errors:")
    for fname, err in files_with_errors:
        print(f"  - {fname}: {err}")
print(f"{'='*60}\n")

# ESCRIURE RESULTATS
with open(RESULTS_FILE, "w") as f:
    f.write(f"# TEST_002 Results - Subfolder {SUBFOLDER}\n\n")
    f.write(f"*Generated from {files_processed} output files*\n\n")
    
    # =========================================================================
    # EXPERIMENT 2.1: Strong Scaling (Processos)
    # =========================================================================
    f.write("## Experiment 2.1: Strong Scaling (Processos)\n\n")
    f.write("*Configuration: 10 epochs, 135 neurons, 1 node*\n\n")
    f.write("| Tasks | Train (s) | Test (s) | Total (s) | Speedup | Efficiency (%) |\n")
    f.write("|-------|-----------|----------|-----------|---------|----------------|\n")
    
    # Buscar tots els tasks amb epochs=10, neurons=135, nodes=1
    exp21_data = {}
    for tasks in sorted(dataset[10].keys()):
        if 135 in dataset[10][tasks][1]:
            data = dataset[10][tasks][1][135]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                exp21_data[tasks] = (train_avg, test_avg, total_avg)
    
    # Calcular speedup (relació amb el més lent)
    if exp21_data:
        baseline_time = max(d[2] for d in exp21_data.values())  # El temps més lent
        
        for tasks in sorted(exp21_data.keys()):
            train_avg, test_avg, total_avg = exp21_data[tasks]
            speedup = baseline_time / total_avg if total_avg > 0 else 0
            efficiency = (speedup / tasks) * 100 if tasks > 0 else 0
            
            f.write(f"| {tasks:5d} | {train_avg:9.4f} | {test_avg:8.4f} | {total_avg:9.4f} | {speedup:7.2f} | {efficiency:14.1f} |\n")
    else:
        f.write("| *No data available* |\n")
    
    f.write("\n")
    
    # =========================================================================
    # EXPERIMENT 2.2: Escalabilitat per Èpoques
    # =========================================================================
    f.write("## Experiment 2.2: Escalabilitat per Èpoques\n\n")
    f.write("*Configuration: 8 tasks, 135 neurons, 1 node*\n\n")
    f.write("| Epochs | Train (s) | Test (s) | Total (s) | Accuracy |\n")
    f.write("|--------|-----------|----------|-----------|----------|\n")
    
    # Buscar tots els epochs amb tasks=8, neurons=135, nodes=1
    exp22_found = False
    for epochs in sorted(dataset.keys()):
        if 8 in dataset[epochs] and 1 in dataset[epochs][8] and 135 in dataset[epochs][8][1]:
            data = dataset[epochs][8][1][135]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                acc_list = [d[3] for d in data if d[3] is not None]
                acc_avg = sum(acc_list) / len(acc_list) if acc_list else 0
                
                f.write(f"| {epochs:6d} | {train_avg:9.4f} | {test_avg:8.4f} | {total_avg:9.4f} | {acc_avg:8.0f} |\n")
                exp22_found = True
    
    if not exp22_found:
        f.write("| *No data available* |\n")
    
    f.write("\n")
    
    # =========================================================================
    # EXPERIMENT 2.3: Escalabilitat per Neurones
    # =========================================================================
    f.write("## Experiment 2.3: Escalabilitat per Neurones\n\n")
    f.write("*Configuration: 8 tasks, 10 epochs, 1 node*\n\n")
    f.write("| Neurons | Train (s) | Test (s) | Total (s) | Accuracy |\n")
    f.write("|---------|-----------|----------|-----------|----------|\n")
    
    # Buscar tots els neurons amb tasks=8, epochs=10, nodes=1
    exp23_found = False
    if 8 in dataset[10] and 1 in dataset[10][8]:
        for neurons in sorted(dataset[10][8][1].keys()):
            data = dataset[10][8][1][neurons]
            if data:
                train_avg = sum(d[0] for d in data) / len(data)
                test_avg = sum(d[1] for d in data) / len(data)
                total_avg = sum(d[2] for d in data) / len(data)
                acc_list = [d[3] for d in data if d[3] is not None]
                acc_avg = sum(acc_list) / len(acc_list) if acc_list else 0
                
                f.write(f"| {neurons:7d} | {train_avg:9.4f} | {test_avg:8.4f} | {total_avg:9.4f} | {acc_avg:8.0f} |\n")
                exp23_found = True
    
    if not exp23_found:
        f.write("| *No data available* |\n")
    
    f.write("\n")
    
    # =========================================================================
    # EXPERIMENT 2.4: Weak Scaling (Nodes)
    # =========================================================================
    f.write("## Experiment 2.4: Weak Scaling (Nodes)\n\n")
    f.write("*Configuration: 8 tasks per node, 10 epochs, 135 neurons*\n\n")
    f.write("| Nodes | Total Tasks | Train (s) | Test (s) | Total (s) | Speedup |\n")
    f.write("|-------|-------------|-----------|----------|-----------|----------|\n")
    
    # Buscar tots els nodes amb epochs=10, neurons=135
    exp24_data = {}
    for tasks in sorted(dataset[10].keys()):
        for nodes in sorted(dataset[10][tasks].keys()):
            if 135 in dataset[10][tasks][nodes]:
                # Verificar que tasks = nodes * 8 (aproximadament)
                if abs(tasks - nodes * 8) <= 1:  # Permetre petita tolerància
                    data = dataset[10][tasks][nodes][135]
                    if data:
                        train_avg = sum(d[0] for d in data) / len(data)
                        test_avg = sum(d[1] for d in data) / len(data)
                        total_avg = sum(d[2] for d in data) / len(data)
                        exp24_data[nodes] = (tasks, train_avg, test_avg, total_avg)
    
    # Calcular speedup (relació amb 1 node)
    if exp24_data:
        baseline_node_time = exp24_data[min(exp24_data.keys())][3] if exp24_data else None
        
        for nodes in sorted(exp24_data.keys()):
            tasks, train_avg, test_avg, total_avg = exp24_data[nodes]
            speedup = baseline_node_time / total_avg if baseline_node_time and total_avg > 0 else 0
            
            f.write(f"| {nodes:5d} | {tasks:11d} | {train_avg:9.4f} | {test_avg:8.4f} | {total_avg:9.4f} | {speedup:8.2f} |\n")
    else:
        f.write("| *No data available* |\n")
    
    f.write("\n")
    f.write("---\n\n")
    
    # RESUM FINAL
    f.write("## Summary\n\n")
    f.write(f"- Total configurations analyzed: {files_processed}\n")
    f.write(f"- Epochs tested: {sorted(set(dataset.keys()))}\n")
    
    all_tasks = set()
    for e in dataset.values():
        all_tasks.update(e.keys())
    f.write(f"- Tasks tested: {sorted(all_tasks)}\n")
    
    all_neurons = set()
    for e in dataset.values():
        for t in e.values():
            for n in t.values():
                all_neurons.update(n.keys())
    f.write(f"- Neurons tested: {sorted(all_neurons)}\n")
    
    all_nodes = set()
    for e in dataset.values():
        for t in e.values():
            all_nodes.update(t.keys())
    f.write(f"- Nodes tested: {sorted(all_nodes)}\n")

print(f"\n{'='*60}")
print(f"✓ Results written to '{RESULTS_FILE}'")
print(f"{'='*60}\n")

# Mostrar preview dels resultats
print("Preview of results:")
with open(RESULTS_FILE, 'r') as f:
    lines = f.readlines()
    for line in lines[:30]:  # Primeres 30 línies
        print(line, end='')
    if len(lines) > 30:
        print("\n... (see full results in results.md)")