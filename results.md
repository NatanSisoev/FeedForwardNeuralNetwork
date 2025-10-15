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

so for each parallelization we have to time the exact portion of the code with and without parallelization, so we used `omp_get_wtime`, for example:

```C
void feed_input(int i) {
    double start = omp_get_wtime();
    #if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT)
    #pragma omp parallel for  // training.feed_input
    #endif
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];
    double elapsed = omp_get_wtime() - start;
    printf("%.16f\n", elapsed);
}
```

add a time right before parallelization, and then the end right after

since these functions are being called thousands of times, it wouldn't make sense to leave the timers there for the general program, so we created a new version of `training.c` where we added and removed the times for each test

for each tag (7 in total) we have to time the parallelizable portion with parallelization enabled and disabled

once we have the average time of the parallelizable part, we can compute the average speedup, here are the results:

[RESULTS](TESTS/TEST_004/results.md)

```
Tag speedups (best first):
TRAINING_FORWARD_PROP_LAYERS -> 2.414226
TRAINING_BACK_PROP_HIDDEN_LAYERS -> 2.103217
TRAINING_UPDATE_WEIGHTS_WEIGHTS -> 1.352390
TRAINING_UPDATE_WEIGHTS_BIASES -> 0.068128
TRAINING_BACK_PROP_OUTPUT_LAYER -> 0.027382
FEED_INPUT -> 0.025384
TRAINING_BACK_PROP_ERRORS -> 0.012050
```

the best one is TRAINING_FORWARD_PROP_LAYERS which doubles the speed

the worst one is TRAINING_BACK_PROP_ERRORS which is 83 times slower than sequential

with this information in mind, we can over-optimize the parallelization by only including the tags that DO improve speed, which are

- TRAINING_FORWARD_PROP_LAYERS      :   2.414226
- TRAINING_BACK_PROP_HIDDEN_LAYERS  :   2.103217
- TRAINING_UPDATE_WEIGHTS_WEIGHTS   :   1.352390

in the following test we will calculate the speedup using only these three tags

## TEST_005

> calculate the speedup with helpful parallelization only:), which are:
> - TRAINING_FORWARD_PROP_LAYERS
> - TRAINING_BACK_PROP_HIDDEN_LAYERS
> - TRAINING_UPDATE_WEIGHTS_WEIGHTS

for this test we will use the times just like in the [previous test](#test_004)

we dont want other parts of the program to play a role, so we will time the minimum part of the code that includes all 3 parallelizations, which is these three lines in `train_neural_net()`:

````C
for (int i = 0; i < num_training_patterns; i++) {
    int p = ranpat[i];

    feed_input(p);
        
    double start = omp_get_wtime();
        forward_prop();
        back_prop(p);
        update_weights();
    double elapsed = omp_get_wtime() - start;
    printf("%.16f\n", elapsed);
}
```

we will average the times with and without parallelization and we will see the speedup (of that part)

[RESULTS](TESTS/TEST_005/results.md)

only with these three tags we have `1.115853` speedup, weird

going back to the for best tags found from [TEST_001](#test_001)

- TRAINING_FORWARD_PROP_LAYERS
- TRAINING_BACK_PROP_ERRORS
- TRAINING_BACK_PROP_HIDDEN_LAYERS
- TRAINING_UPDATE_WEIGHTS_WEIGHTS

we achieve a speedup of `5.039654` with a total of around `2.084183` seconds (including printing)

