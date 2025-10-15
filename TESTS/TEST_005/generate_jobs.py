import itertools

flags = [
    "TRAINING_FORWARD_PROP_LAYERS",
    "TRAINING_BACK_PROP_OUTPUT_LAYER",
    "TRAINING_UPDATE_WEIGHTS_WEIGHTS",
]


print(f"sbatch TESTS/TEST_005/TEST_005.sub {','.join(flags)}")
print(f"sbatch TESTS/TEST_005/TEST_005.sub X")

# python3 TESTS/TEST_005/generate_jobs.py > TESTS/TEST_005/jobs.sh
# bash TESTS/TEST_005/jobs.sh
# squeue -u $USER

# CAUTION: it considers all files in OUT, so if you run multiple tests, results will be aggregated
# echo -e "\n"\# RESULTS \($(date "+%Y-%m-%d %H:%M:%S")\) >> TESTS/TEST_005/results.md && echo \`\`\` >> TESTS/TEST_005/results.md && python3 TESTS/TEST_005/analysis.py >> TESTS/TEST_005/results.md && echo \`\`\` >> TESTS/TEST_005/results.md 
