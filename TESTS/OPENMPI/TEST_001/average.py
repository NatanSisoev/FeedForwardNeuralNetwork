import re
import sys
from statistics import mean

# Fitxer d'entrada
filename = sys.argv[1] if len(sys.argv) > 1 else "./OUT/A/out_76672.out"

# Dades a extreure
fields = ["FEED_INPUT", "FORWARD_PROP", "BACK_PROP", "UPDATE_WEIGHTS",
          "TRAIN_TIME", "TEST_TIME", "TOTAL_TIME"]

# Diccionari per guardar totes les dades
data = {f: [] for f in fields}

with open(filename) as f:
    content = f.read()

# Troba tots els blocs #START ... #END
blocks = re.findall(r"#START:\d+(.+?)#END:\d+", content, re.DOTALL)

for block in blocks:
    for field in fields:
        # Cerca la línia corresponent al camp
        match = re.search(rf"{field}[:\t ]+([0-9.]+)", block)
        if match:
            data[field].append(float(match.group(1)))

# Calcula mitjanes
print(f"Mitjanes per {filename}:")
for field in fields:
    values = data[field]
    if values:
        print(f"{field:15s}: {mean(values):.6f}")
