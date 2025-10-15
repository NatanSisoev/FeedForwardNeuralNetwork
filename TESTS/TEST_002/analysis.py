import os
import glob

# Folder containing all .out files
ROOT_DIR = "TESTS/TEST_002/OUT"

def read_out_file(file_path):
    """
    Reads a .out file.
    Returns (tags, time) if valid, None if file has errors.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
        # print("Read lines:", lines)
        if len(lines) < 3:
            return None  # incomplete file

        try:
            nums = lines[2].strip().split("\t")
            num_encerts = int(nums[0])
            tags = lines[1].strip()
            time = float(nums[1])
        except ValueError:
            print("ERROR: Value conversion error in file:", file_path)
            # If conversion fails, assume error messages exist
            return None

        if num_encerts != 885:
            print("ERROR: Invalid number of encerts:", num_encerts, "in file", file_path)
            return None  # skip invalid encerts

        return tags, time

def main():
    results = []
    # Recursively find all .out files
    for filepath in glob.glob(os.path.join(ROOT_DIR, "*.out")):
        # print("Processing file:", filepath)
        data = read_out_file(filepath)
        if data:
            results.append(data)

    if not results:
        print("No valid data found.")
        return

    # Sort by time ascending
    results.sort(key=lambda x: x[1])
    best_tags, best_time = results[0]
    print(f"Best tag combination: {best_tags}")
    print(f"Smallest time: {best_time}")
    print("\nTop 5 tag combinations by time:")
    for tags, t in results[:5]:
        print(f"{tags} -> {t}")

if __name__ == "__main__":
    main()
