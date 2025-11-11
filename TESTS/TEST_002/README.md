# TEST_002

## Objective

The objective of this test is to determine the individual speedup of each possible parallelization. The idea is that, as explained in the objective of the first test, parallelization is not always optimal if the task is small in itself.

With this test, we will be able to determine each individual speedup and see if they're coherent with the results from [TEST_001](../TEST_001/test.md). Hopefully, we will see that the tags that ranked higher will have speedup greater than one, and those tags that have a speedup lower than one shouldn't appear in the top ranking.

## Methodology

Since the server has so much varaibility, we will only time the minimal part of the code that includes the parallelization. We will use the `OMP` function `omp_get_wtime()` to time each `for` with and without parallelization and then calculate the speedup.

We do not want to change the existing files, we want to make this test reproducible without having to alter the codebase each time. Therefore, we will copy the `training/training.c` file to this folder, [training.c](training.c), and we will pass a flag to the `scheduler.sub` to consider this file instead of the main one.

Take, for example, the `FEED_INPUT` tag. We would time this parallelization in the following manner:

```C
void feed_input(int i) {
    double start_time = omp_get_wtime();
    #if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT)
    #pragma omp parallel for  // training.feed_input
    #endif
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];
    printf("FI: %18.16f\n", omp_get_wtime() - start_time);
}
```

From the output, we are able to sum up all the times and determine the speedups. For now, efficiency is not relevant (just divide by 12), since it becomes important only when comparing to different number of threads, which we will do in the following tests. From there we will see which ones are worth including in the final version of the code.

## Execution

As before, we've created a `Python` [script](run.py) that, this time, simply runs the code with and without all the tags.

After execution, it interprets the results and saves the relevant metrics, such as speedup and efficiency.

It can be executed from the root folder using [this](../../run_test.py) other script with the following command:

```bash
python3 run_test.py 002 [SUBFOLDER, MODE]
```

Available options are:

- `SUBFOLDER`: subfolder inside `OUT/` to output `.out` files` (the name of the test)
- `MODE`: either `e` to execute the test, `a` to analyze the results, or both

No arguments at all will create a new subfolder named with the first unused upper case letter, where you will find all the `.out` files. Results will be available in [this](results.md) file under the subfolder's header.

## Results

The first run (test `A`) gives the following results:

|  #  | time_par   | fraction_par | time_seq   | fraction_seq | speedup  | tag  |
|-----|------------|--------------|------------|--------------|----------|------|
|  1  | 0.359971   | 0.116600     | 3.158224   | 0.304900     | 8.7736   | `BPH`|
|  2  | 0.335285   | 0.108600     | 2.638832   | 0.254800     | 7.8704   | `UWW`|
|  3  | 1.123978   | 0.364000     | 4.482735   | 0.432800     | 3.9883   | `FPL`|
|  4  | 0.091257   | 0.029600     | 0.016294   | 0.001600     | 0.1786   | `FI` |
|  5  | 0.185019   | 0.059900     | 0.024912   | 0.002400     | 0.1346   | `UWB`|
|  6  | 0.902893   | 0.292400     | 0.033675   | 0.003300     | 0.0373   | `BPO`|
|  7  | 0.089107   | 0.028900     | 0.003167   | 0.000300     | 0.0355   | `BPE`|

We repeat the same test 4 more times (tests `B`, `C`, `D` and `E`) and we get the following averages across all 5 tests:

| #   | time_par | fraction_par | time_seq | fraction_seq | speedup | tag |
|-----|----------|--------------|----------|--------------|---------|-----|
| 1 | 0.331066 | 0.1052       | 3.156015 | 0.3053       | 9.5527  | `BPH` |
| 2 | 0.347434 | 0.1103       | 2.622788 | 0.2538       | 7.5549  | `UWW` |
| 3 | 1.124091 | 0.3569       | 4.479600 | 0.4334       | 3.9865  | `FPL` |
| 4 | 0.101662 | 0.0323       | 0.016016 | 0.0016       | 0.1584  | `FI`  |
| 5 | 0.195332 | 0.0620       | 0.024630 | 0.0024       | 0.1264  | `UWB` |
| 6 | 0.958994 | 0.3042       | 0.033569 | 0.0033       | 0.0351  | `BPO` |
| 7 | 0.092299 | 0.0293       | 0.002868 | 0.0003       | 0.0312  | `BPE` |

We notice that the only tags that have a speedup greater than one are `TRAINING_BACK_PROP_HIDDEN_LAYERS`, `TRAINING_UPDATE_WEIGHTS_WEIGHTS` and `TRAINING_FORWARD_PROP_LAYERS`, which are exactly the ones with the highest frequency in the top rankings form [TEST_001](../TEST_001/test.md).

Certainly, this is not coincidental. It makes sense for the the tags that have the best speedup to rank among the highest combinations.

In fact, we observe that for all the other tags, not only do they not improve the execution time, they worsen it. Their lower than 1 (even 0.2) speedup shows that the time with parallelization is at least 5 times longer than without.

So it might come as a surprise to see that in the prvious rankings we still had tags like `BPE` show up in the top-performing combinations. If we look carefully at all the data, we see that all these tags with bad speedup take up less than 1% altogether. So even if they get slower, server variability eats up all that variation either way.

We could compute the efficiency say, for the best tag: 0.80, which is very close to the theoretical one.

## Conclusions

With this test we succesfully and confidently discard the following parallelizations: `FEED_INPUT`, `TRAINING_UPDATE_WEIGHTS_BIASES`, `TRAINING_BACK_PROP_OUTPUT_LAYER` and `TRAINING_BACK_PROP_ERRORS`. We have seen that these parallelizations make the program slower, having a speedup considerably less than 1.

From now on, we will label the three best performing tags `TRAINING_BACK_PROP_HIDDEN_LAYERS`, `TRAINING_UPDATE_WEIGHTS_WEIGHTS` and `TRAINING_FORWARD_PROP_LAYERS` as `OPT`. This will also be a flag for the compiler, which we will use for our next tests (and consider our final version up to this test).
