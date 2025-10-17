# Explanation

## tags

- `ALL`
    - `TR`: `TRAINING`
        - `FI`: `FEED_INPUT`
        - `FP`: `FORWARD_PROP`
            - `FPL`: `TRAINING_FORWARD_PROP_LAYERS`
        - `BP`: `BACK_PROP`
            - `BPE`: `TRAINING_BACK_PROP_ERRORS`
            - `BPO`: `TRAINING_BACK_PROP_OUTPUT_LAYER`
            - `BPH`: `TRAINING_BACK_PROP_HIDDEN_LAYERS`
        - `UW`: `UPDATE_WEIGHTS`
            - `UWW`: `TRAINING_UPDATE_WEIGHTS_WEIGHTS`
            - `UWB`: `TRAINING_UPDATE_WEIGHTS_BIASES`

## `run_test.py`

The test runner first takes in the number of the test (3 digits left zero padding), and then as many arguments as needed, that will later pass to the chosen `run.py` file. The arguments for each script are exaplined at the `test.md` of each test.

## `scheduler.sub`

The scheduler takes in 6 arguments:

1. test number: 3 digits padded with 0's to the left (e.g. 007)
2. flags: all uppercase, separeted by a comma ","
3. repetitions: number of times to execute the program
4. subfolder: specific subfolder inside `OUT/`
5. include option for training file to resolve dependencies (relevant for `TEST_002`)
6. file path of the `training.c` file (relevant for `TEST_002`)

