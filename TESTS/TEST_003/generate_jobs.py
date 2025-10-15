import itertools

flags = [
    "TRAINING_UPDATE_WEIGHTS_BIASES",
    "TRAINING_UPDATE_WEIGHTS_WEIGHTS",
]

REPEAT = 10  # number of repetitions for each configuration

# Generate all non-empty combinations
for r in range(len(flags) + 1):
    for combo in itertools.combinations(flags, r):
        # Join flags with commas for passing to sbatch
        flag_str = ",".join(combo)
        print(f"sbatch TESTS/TEST_003/TEST_003.sub {flag_str} {REPEAT}")

print("mkdir -p TESTS/TEST_003/OUT")  # create OUT directory if it doesn't exist

# python3 TESTS/TEST_003/generate_jobs.py > TESTS/TEST_003/jobs.sh
# bash TESTS/TEST_003/jobs.sh

# CAUTION: it considers all files in OUT, so if you run multiple tests, results will be aggregated
# echo -e "\n"\# RESULTS \($(date "+%Y-%m-%d %H:%M:%S")\) >> TESTS/TEST_003/results.md && echo \`\`\` >> TESTS/TEST_003/results.md && python3 TESTS/TEST_003/analysis.py >> TESTS/TEST_003/results.md && echo \`\`\` >> TESTS/TEST_003/results.md 
