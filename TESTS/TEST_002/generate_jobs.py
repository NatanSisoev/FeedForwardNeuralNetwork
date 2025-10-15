import itertools

flags = [
    "FEED_INPUT",
    "FORWARD_PROP",
     "BACK_PROP",
     "UPDATE_WEIGHTS",
]

# Generate all non-empty combinations
for r in range(len(flags) + 1):
    for combo in itertools.combinations(flags, r):
        # Join flags with commas for passing to sbatch
        flag_str = ",".join(combo)
        print(f"sbatch TESTS/TEST_002/TEST_002.sub {flag_str}")

print("mkdir -p TESTS/TEST_002/OUT")  # create OUT directory if it doesn't exist

# python3 TESTS/TEST_002/generate_jobs.py > TESTS/TEST_002/jobs.sh
# bash TESTS/TEST_002/jobs.sh

# CAUTION: it considers all files in OUT, so if you run multiple tests, results will be aggregated
# echo -e "\n"\# RESULTS \($(date "+%Y-%m-%d %H:%M:%S")\) >> TESTS/TEST_002/results.md && echo \`\`\` >> TESTS/TEST_002/results.md && python3 TESTS/TEST_002/analysis.py >> TESTS/TEST_002/results.md && echo \`\`\` >> TESTS/TEST_002/results.md 
