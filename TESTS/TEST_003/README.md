# TEST_003

## Objective

The objective of this test is to test out best version of the code on different settings and configurations, including servers, partition files, number of threads, number of epochs, and number of neurons in the hidden layer.

## Methodology

For this test we will leave the optimal parallelization that we've found and we will vary other configurations as explained in the objective. We will consider the sequential time the time with `OPT` optimization but with only one thread, since it is easier from the programatic point of view to do it this way, but it is also equivalent to no parallelization.

Since we have to switch ssh hosts to change servers, this is a setting that we will have to manually adjust whenever we change hosts. The partition file and number of threads are set in the `scheduler.job`, and the number of epochs and neurons in the hidden layer are found in the configuration files. For each combination, we will have to edit both files and set in out preferences, then revert the files to the original state.

To do this, we will automate the process by changing the `scheduler.sub` for each run and then submitting the job, and for the configuratio file we will create a copy with the desired configurations in `TESTS/TEST_003/configuration` and pass it to the main program as an argument. We have to create a copy because the file is read at runtime, not when the job is submitted, so submitting the job with one version and then changing the file might result in unexpected behaviour. This latter folder can be deleted after the execution of the jobs, it is not important after that, since the configurations are stored in each output file metadata.

## Execution

We've created a [script](run.py) that runs the tests for us. For this test, the default parallelization is `OPT`, although it can be changed in the settings of the script.

To execute the test you can use the general test runner [script](../../run_test.py):

```bash
python3 run_test.py 002 [SUBFOLDER, MODE, VARIATE]
```

Available options are:

- `SUBFOLDER`: subfolder inside `OUT/` to output `.out` files (empty or `X` will create the next following letter)
- `MODE`: either `e` to execute the test, `a` to analyze the results, or both
- `VARIATE`: flags controlling test variations: partitions `P`, number of threads `T`, number of epochs `E` and number of neurons in the hidden layer `N` (or any combination of these, e.g. `PTE`)

Arguments must go in order, so if you want to spcify a test variation flag, you must include the subfolder and the mode.

Results will be available in [this](results.md) file under the subfolder's header.

So, say you want to test all possible number of threads: you have to indicate `T` in the execution and the script will run one execution for each possible thread quantity. If you want to test all number of threads across different partition files, you can indicate both `PT` and the script will iterate over all possible combinations.

## Results

For our first test `A`, we will execute all variations (first in the Wilma server). So the first command we run is

```
python3 run_test.py 003 A ea PTEN
```

This will variate all possible parameters and save the outputs in the `TESTS/TEST_003/OUT/A` folder.

The results are best seen in the results [file](results.md#a), since it includes a lot of tables, containing runtimes, speedups, efficiencies and hits (average over all number of threads). The average number of hits is included because if one execution achieves less hits, it will be shown in the average, so it's a fast way to assure that all hits have the same maximum value.

## Conclusions

Here we will discuss some of the most important findings of this test.

First, we see that the more epochs we include the more efficiency we get from the parallelization. Increasing the number of epochs increases precisely the part of the code that is parallelizable, so it makes sense (according to Gustafson's Law), that the efficiency goes up.

Using `nodo.q` the speedup is around 5 and 8 for 135 and 250 neurons respectively. Again, increasing the neurons increases the parallelizable part, so a better speedup is achieved.

For `new-nodo.q`, we see that the runtimes are basically cut by half, but the speedup is not that amazing, achieving around 3 and 5 for 135 and 250 neurons respectively.

Looking at the efficiencies, we see that pretty high numbers are achieved at lower number of threads, with the fraction going up for higher threads. This is normal since the workload is the same and the sequential part grows in percentatge each time, according to Amdahl's Law.

We see that the number of hits coincide with the reference, and the average is precisely that number, meaning that all executions achieved that number (only in the rare case of two executions perfectly cancelling each other, which hasn't happened, since we inspected the individual hits).
