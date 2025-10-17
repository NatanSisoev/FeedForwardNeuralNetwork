# TEST_002

## Objective

The objective of this test is to determine the individual speedup of each possible parallelization. The idea is that, as explained in the objective of the first test, parallelization is not always optimal if the task is small in itself.

With this test, we will be able to determine each individual speedup and see if they're coherent with the results from [TEST_001](../TEST_001/test.md). Hopefully, we will see that the tags that ranked higher will have speedup greater than one, and those tags that have a speedup lower than one shouldn't appear in the top ranking.

## Methodology

Since the server has so much varaibility, we will only time the minimal part of the code that includes the parallelization. We will use the `OMP` function `omp_get_wtime()` to time each `for` with and without parallelization and then calculate the speedup.

FRom there we will be able to determine which speedups are worth including in the final version of the code.

## Execution

As before, we've created a `Python` [script](run.py) that, this time, simply runs the code with and without all the tags.

After execution, it interprets the results and saves the relevant metrics, such as speedup and efficienciy.

It can be executed from the root folder using [this](../../run_test.py) other script with the following command:

```bash
python3 run_test.py 002 [SUBFOLDER, MODE]
```

Available options are:

- `SUBFOLDER`: subfolder inside `OUT/` to output `.out` files` (the name of the test)
- `MODE`: either `e` to execute the test, `a` to analyze the results, or both

No arguments at all will create a new subfolder named with the first unused upper case letter, where you will find all the `.out` files. Results will be available in [this](results.md) file under the subfolder's header.

## Results


## Conclusions

