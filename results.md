# FLAGS

- ALL: all files
    - TRAINING: training.c
        - FEED_INPUT: feed_input()
            > training.feed_input
        - FORWARD_PROP: forward_prop()
            - TRAINING_FORWARD_PROP_LAYERS
                > training.forward_prop.layers
        - BACK_PROP: back_prop()
            - TRAINING_BACK_PROP_ERRORS
                > training.back_prop.errors
            - TRAINING_BACK_PROP_OUTPUT_LAYER
                > training.back_prop.output_layer
            - TRAINING_BACK_PROP_HIDDEN_LAYERS
                > training.back_prop.hidden_layers
        - UPDATE_WEIGHTS: update_weights()
            - TRAINING_UPDATE_WEIGHTS_WEIGHTS
                > training.update_weights.weights
            - TRAINING_UPDATE_WEIGHTS_BIASES
                > training.update_weights.biases

---

# BEST

1. 2.004625
    
    `TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS`

2. 2.121777

    `TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS UPDATE_WEIGHTS`


---

# TESTS

## TEST_001

> testing all possible parallelization combinations

[RESULTS](TESTS/TEST_001/results.md)

```
Best tag combination: TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS
Smallest time: 2.004625
```

`ERROR: Value conversion error in file`: this is because of some errors in execution, hopefully gone in next tests

## TEST_002

> testing all possible combinations of functions (function group tags)

[RESULTS](TESTS/TEST_002/results.md)

```
Best tag combination: FORWARD_PROP BACK_PROP UPDATE_WEIGHTS
Smallest time: 3.142874
```

`ERROR: Value conversion error in file: TESTS/TEST_002/OUT/TEST_002.sub_63161.out`: still error for tags: 'UPDATE_WEIGHTS'

now i have to check what is the problem with UPDATE_WEIGHTS:

- executed a couple of times, no problems
- no race conditions found
- a bit slower than no parallelization
- conduct TEST_003 for further study

## TEST_003

> determine parallelization for UPDATE_WEIGHTS: run all combinations of tags `TRAINING_UPDATE_WEIGHTS_BIASES` and `TRAINING_UPDATE_WEIGHTS_WEIGHTS` repeating each one `REPEAT` times

[RESULTS](TESTS/TEST_003/results.md)

ran the tests:

- the first run is always ~1 second slower than the following, so we discarded the first run (and other runs with random errors)
- did the average over all runs

`REPEAT = 10`:

```
Average times by tag:
None -> 10.3567965
TRAINING_UPDATE_WEIGHTS_BIASES -> 10.363025375
TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 10.365545000000001
TRAINING_UPDATE_WEIGHTS_BIASES TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 10.372693700000003
```

basically the same time

ran again with 20 repetitions

`REPEAT = 20`:

```
Average times by tag:
TRAINING_UPDATE_WEIGHTS_BIASES -> 11.371065842105262
TRAINING_UPDATE_WEIGHTS_BIASES TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 11.426622299999998
TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 11.43401352631579
None -> 11.897826388888891
```

now all times are up 1 second for no apparent reason

the sequential is not the slowest by big margin ~0.5 seconds

cummulative (~30 iterations):

```
Average times by tag:
TRAINING_UPDATE_WEIGHTS_BIASES -> 10.86704560855263
TRAINING_UPDATE_WEIGHTS_BIASES TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 10.899658
TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 10.899779263157896
None -> 11.127311444444445
```

over the 30 iterations, parallelized has made a difference, so for now we will leave it

very small difference between each tag, we can leave both

## TEST_004

> going back to all tags, now we're interested in the individual tags, which provides the best speedup

