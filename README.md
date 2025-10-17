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


## `scheduler.sub`

The scheduler takes in 4 arguments:

1. test number: 3 digits padded with 0's to the left (e.g. 007)
2. flags: all uppercase, separeted by a comma ","
3. repetitions: number of times to execute the program
4. subfolder: specific subfolder inside `OUT/`

