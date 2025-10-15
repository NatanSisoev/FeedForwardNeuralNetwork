import itertools

flags = [
    "FEED_INPUT",
    "TRAINING_FORWARD_PROP_LAYERS",
    "TRAINING_BACK_PROP_ERRORS",
    "TRAINING_BACK_PROP_OUTPUT_LAYER",
    "TRAINING_BACK_PROP_HIDDEN_LAYERS",
    "TRAINING_UPDATE_WEIGHTS_WEIGHTS",
    "TRAINING_UPDATE_WEIGHTS_BIASES"
]


for flag in flags:
    print(f"sbatch TESTS/TEST_004/TEST_004.sub NO_{flag}")
    print(f"sbatch TESTS/TEST_004/TEST_004.sub {flag}")
    print()

# python3 TESTS/TEST_004/generate_jobs.py > TESTS/TEST_004/jobs.sh
# bash TESTS/TEST_004/jobs.sh
# squeue -u $USER

# CAUTION: it considers all files in OUT, so if you run multiple tests, results will be aggregated
# echo -e "\n"\# RESULTS \($(date "+%Y-%m-%d %H:%M:%S")\) >> TESTS/TEST_004/results.md && echo \`\`\` >> TESTS/TEST_004/results.md && python3 TESTS/TEST_004/analysis.py >> TESTS/TEST_004/results.md && echo \`\`\` >> TESTS/TEST_004/results.md 
