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

# Generate all non-empty combinations
for r in range(len(flags) + 1):
    for combo in itertools.combinations(flags, r):
        # Join flags with commas for passing to sbatch
        flag_str = ",".join(combo)
        print(f"sbatch job.sub {flag_str}")


# python3 run_all_flag_combinations.py > all_jobs.sh
# bash all_jobs.sh