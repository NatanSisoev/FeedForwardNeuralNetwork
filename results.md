# FLAGS

- ALL: all files
    - TRAINING: training.c
        - FEED_INPUT: feed_input()
            - `training.feed_input`
        - FORWARD_PROP: forward_prop()
            - TRAINING_FORWARD_PROP_LAYERS
                - `training.forward_prop.layers`
        - BACK_PROP: back_prop()
            - TRAINING_BACK_PROP_ERRORS
                - `training.back_prop.errors`
            - TRAINING_BACK_PROP_OUTPUT_LAYER
                - `training.back_prop.output_layer`
            - TRAINING_BACK_PROP_HIDDEN_LAYERS
                - `training.back_prop.hidden_layers`
        - UPDATE_WEIGHTS: update_weights()
            - TRAINING_UPDATE_WEIGHTS_WEIGHTS
                - `training.update_weights.weights`
            - TRAINING_UPDATE_WEIGHTS_BIASES
                - `training.update_weights.biases`

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

```C
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

only with these three tags we have `5.471035` speedup, and total time `2.000039` including printing, pretty good

going back to the for best tags found from [TEST_001](#test_001)

- TRAINING_FORWARD_PROP_LAYERS
- TRAINING_BACK_PROP_ERRORS
- TRAINING_BACK_PROP_HIDDEN_LAYERS
- TRAINING_UPDATE_WEIGHTS_WEIGHTS

we achieve a speedup of `5.039654` with a total of around `2.084183` seconds (including printing)

so we see that removing the tag that wasn't helping (`TRAINING_BACK_PROP_ERRORS` had worse speedup < 1) increases the speedup (from `5.039654` to `5.471035`)

## TEST_006

> going back to the results from [TEST_001](#test_001) to test the average over `N` runs

so we consider the top 5 best combinations from the first test and repeat them `N` times

the top 5 was:

```
1. 2.004625
    - TRAINING_FORWARD_PROP_LAYERS
    - TRAINING_BACK_PROP_HIDDEN_LAYERS
    - TRAINING_UPDATE_WEIGHTS_WEIGHTS
    - TRAINING_BACK_PROP_ERRORS
2. 2.049836
    - TRAINING_FORWARD_PROP_LAYERS
    - TRAINING_BACK_PROP_HIDDEN_LAYERS
    - TRAINING_UPDATE_WEIGHTS_WEIGHTS
3. 2.142863
    - TRAINING_FORWARD_PROP_LAYERS
    - TRAINING_BACK_PROP_HIDDEN_LAYERS
    - TRAINING_UPDATE_WEIGHTS_WEIGHTS
    - FEED_INPUT
4. 2.148185
    - TRAINING_FORWARD_PROP_LAYERS
    - TRAINING_BACK_PROP_HIDDEN_LAYERS
    - TRAINING_UPDATE_WEIGHTS_WEIGHTS
    - TRAINING_BACK_PROP_ERRORS
    - FEED_INPUT
5. 2.227651
    - TRAINING_FORWARD_PROP_LAYERS
    - TRAINING_BACK_PROP_HIDDEN_LAYERS
    - TRAINING_UPDATE_WEIGHTS_WEIGHTS
    - TRAINING_UPDATE_WEIGHTS_BIASES
```

a couple of observations:

- all 5 combinations include the set `TRAINING_FORWARD_PROP_LAYERS`, `TRAINING_BACK_PROP_HIDDEN_LAYERS`, `TRAINING_UPDATE_WEIGHTS_WEIGHTS` which we name `OPTIMAL`
- `OPT` set coincides with the parallelization that have speedup > 1 from [TEST_004](#test_004)
- number 1 and 4 include `TRAINING_BACK_PROP_ERRORS` which has a speedup of `0.012050`, the worst of all of them
- number 2 includes only `OPT`
- number 3 and 4 include `FEED_INPUT` with a speedup of `0.025384`
- number 5 includes `TRAINING_UPDATE_WEIGHTS_BIASES` with a speedup of `0.068128`

it might be confusing that including parallelizations that slow down the execution time (speedup < 1) decreases the overall time (like number 1), but we believe it is a matter of variability, and repeating the test `N` times will average out the results

same as the previous test, we will isoalte the minimal part of the code that uses all parallelization, which is this part (this time including `feed_input(p)`):

```C
for (int i = 0; i < num_training_patterns; i++) {
    int p = ranpat[i];

    double start = omp_get_wtime();
        feed_input(p);
        forward_prop();
        back_prop(p);
        update_weights();
    double elapsed = omp_get_wtime() - start;
    printf("%.16f\n", elapsed);
}
```
we will submit 1 job for each tag combination at the same time, then run this multiple times, so to minimize the effects of the server variability: if we first run one combination 10 times then the others, there will be more difference between each combination, whereas if we run them once one after the other and then repeat, there might be more variability between runs but that's ok since it's going to be more or less the same for each tag combination

[RESULTS](TESTS/TEST_006/results.md)

from just 1 run we see that only the 3 tags from `OPT` outperform the others by a big margin, achieving a speedup of `6.082449` compared to the second best, `5.399156`, but it's only one run, so it's not that significant

after 10 runs, we get these results:

```
OPT -> speedup 5.139883
OPT + TRAINING_BACK_PROP_ERRORS -> speedup 4.979350
OPT + TRAINING_BACK_PROP_ERRORS + FEED_INPUT -> speedup 4.734256
OPT + TRAINING_UPDATE_WEIGHTS_BIASES -> speedup 4.562877
OPT + FEED_INPUT -> speedup 4.455098
```

small difference, but over 10 runs we are pretty certain it is somewhat significant

## TEST_007

> for this test we are interested in how the execution time is spread across tasks and various parts of the program

specifically, we want to figure out how much of the time is spent on each of the following parts (percent-wise):

- initialization
- training
- testing
- deconstruction

the initialization and deconstruction of the variables are done outside the timer that is already set, so we will only consider testing and training, which we can split into:

- training
    - `loading_patterns`
    - `pattern_shuffle`
    - for each pattern:
        - `input_feed`
        - `forward_propagation`
        - `backward_propagation`
        - `weight_update`
- testing
    - `loading_patterns`
    - for each pattern:
        - `predicting`
        - `recognition`


we print the `omp_get_wtime()` at each place and then calculate the times for each part

[RESULTS](TESTS/TEST_007/results.md)

these are the results over 10 executinos:

```
forward_propagation      :  4.283941 s ( 42.71%)
weight_update            :  2.686664 s ( 26.78%)
backward_propagation     :  2.658612 s ( 26.50%)
predicting               :  0.209569 s (  2.09%)
loading_patterns         :  0.056804 s (  0.57%)
input_feed               :  0.015834 s (  0.16%)
pattern_shuffle          :  0.000505 s (  0.01%)
recognition              :  0.000095 s (  0.00%)
```

so we see that `backward_propagation`, `weight_update` and `input_feed` take the majority of time


let's now execute the same program with `OPT` tags toggled and see how that changes the percentatges

notice that parallelization only happens for the parts `forward_propagation`, `backward_propagation` and `weight_update`, so these should drop in percentatge, following amdahls law

```
forward_propagation      :  1.040508 s ( 50.84%)
backward_propagation     :  0.423518 s ( 20.69%)
weight_update            :  0.394067 s ( 19.26%)
predicting               :  0.050894 s (  2.49%)
loading_patterns         :  0.037077 s (  1.81%)
input_feed               :  0.033087 s (  1.62%)
pattern_shuffle          :  0.000501 s (  0.02%)
recognition              :  0.000154 s (  0.01%)
```

we see how the parallelized parts are by far the ones that drop in time

lets look at the speedups of the three parallelized parts

| Phase                  | Original (s) | OPT (s)  | Speedup |
|------------------------|-------------|----------|---------|
| forward_propagation     | 4.283941    | 1.040508 | 4.12x   |
| backward_propagation    | 2.658612    | 0.423518 | 6.28x   |
| weight_update           | 2.686664    | 0.394067 | 6.82x   |

indeed, we see that they offer a considerable speedup to the overall program, covering more than 90% of the work done (in sequential)

we see that `backward_propagation` and `weight_update` drop in percentatge, as they should since now the non-parallelized part of the program takes more space percentually

nonehtless, `forward_propagation` grows in percentatge: we believe that is due to its speedup being not as high as the other two, so although it should go down in percentatge it doesnt because compared to the others it doesnt drop as much

conclusions:

- we could parallelize the predicting and recognition part but that would, at best, drop the total time `0.01` seconds, which is orders of magnitude lower than the server execution time variability, so in this speciric scenario, with the current size of the neural netweork it makes no sense to further parallelize any other parts (iteration-wise)
- we think we have parallelized all the parts that were parallelizable at the moment, with the current structure and working of the code
- further parallelization would be possible if we were allowed to change the structure of the code, like for example "true" pseudo-random initialization for the weights, parallel training then aggregation, and other things

---
