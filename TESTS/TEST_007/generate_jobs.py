import itertools

# Number of repetitions
N = 1

print(f"sbatch TESTS/TEST_007/TEST_007.sub {N}")

# python3 TESTS/TEST_007/generate_jobs.py > TESTS/TEST_007/jobs.sh
# bash TESTS/TEST_007/jobs.sh

# CAUTION: it considers all files in OUT, so if you run multiple tests, results will be aggregated
# python3 TESTS/TEST_007/analysis.py
# echo -e "\n"\# RESULTS \($(date "+%Y-%m-%d %H:%M:%S")\) >> TESTS/TEST_007/results.md && echo \`\`\` >> TESTS/TEST_007/results.md && python3 TESTS/TEST_007/analysis.py >> TESTS/TEST_007/results.md && echo \`\`\` >> TESTS/TEST_007/results.md 
