# BEST TIME

`OPT`: 1.916558 [test_66433.out](TESTS/TEST_001/OUT/E/test_66433.out)

# Execution

For a fast compilation of the code, just run:

```bash
sbatch job-omp.sub
```

This already has the necessary options for our final version of the parallelized code.

ATTENTION: If you want to compile it independently, include the `-DOPT` flag to the compiler:

```
gcc -DOPT -fopenmp main.c ...
```

Otherwise, it will run the sequential version. You can "turn on" any optimization by defining the flag name, using [these](#tags) tags.

You can make use of the [scheduler](#scheduler) for more advances runs, or even out custom testing scripts explained [here](#tests).

# Preliminaries

Loops/tasks: we've seen that the program is quite straighforward sequential task-wise, so we have not spotted any options for paralellelization task-wise. As we will discuss in the following point, the most demanding part of the code (which is training) is the one we have paralellized and it's the only one that really matters.

What loops: only training is worth parallelizing. Testing takes a very small portion of the full program, and part of it (which is forward propagation) is already parallelized because it is a function included in the training process. The only part maybe worth giving a shot is copying the image to the first layer. We observe that this for loop is exactly the same as the `feed_input` one, so we will treat them equally: if `feed_input` has good speedup, we will also include the for from testing, if not we will not include it. So for the main tests we will only consider the functions from the `training.c` file.

Per thread analysis: considering that the parallelization that we will do is loop-wise, we've come tot the conclusion that it is not necessary to carry out a core-level analysis for the runtime. For timing we will use the native `omp_get_wtime()` function directly from `OpenMP`, which is the most fine-graned tool we can use for precise masures. Any other tools like `perf` or `likwid` are not helpful in our case since they always include the whole program, and we are only interested in bits of it. We have also notices that all the parallelizable for's in the training phase have distributed load, so the workload per iteration is exactly equal for all repetitions, so there's no need for specific scheduling or per-thread analysis.

Server and partition: for all of our analysis and execution we have used the Wilma server. In the first two tests, we use the default `nodo.q` partition, and in the third one we compare both partition files.

Testing: we have automated the testing of our code with Python scripts. They automatically generate all the necessary jobs depending on the testing we are carrying out, and they automatically create the jobs, gather the data and analyze the results. Because of this, we are able to schedule very long tasks (the longest was around 4 hours) that we can leave running overnight.

Jobs: we have created a personalised `scheduler.sub` that satisfies all the necessities of the tests we've designed. It takes in several arguments and is very useful for fast testing of the script.

GitHub: we have documented all of the process we've done using `GitHub`, organizing the work in differnt branches and commits.

# TESTS

- [TEST_001](TESTS/TEST_001/test.md)
- [TEST_002](TESTS/TEST_002/test.md)
- [TEST_003](TESTS/TEST_003/test.md)

# Explanation

## Tags

While we were thinking of the ways to test our parallelizations, we came up with a clever way to enable and disable them without having to modify the codebase each time.

For clarity and explanation we've have named each parallelization according to the task it was parallelizing, to which we will refer as tags or flags:

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

They appear in hierarchical structure because parallelizing the weights updates implies parallelizing both weights and biases. We have also included a shortname for each one.

To enable and disable them from the terminal, we've wrapped each `pragma` comment in `#if-#endif` clauses with check if that specific flag is defined. Take for example the tag `FEED_INPUT`:

```c
#if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT)
#pragma omp parallel for  // training.feed_input
#endif
for (int j = 0; j < num_neurons[0]; j++)
    lay[0].actv[j] = input[i][j];
```

It checks if that tag or any of its parents is defined, and if so, it enables the `OpenMP` directive. If none of the flags is given, it skips the directive, making the code run sequentially.


## Tests

For the analysis of our parallelization, we've thought of a particular structure for testing and storing up results. In the main directory we have the `TESTS` folder, which contains all tests.

We consider test the following folder and file structure:

```
TEST_X
├── OUT
│   ├── A
│   ├── ...
│   └── E
├── README.md
├── results.md
└── run.py
```

where `X` is the number of the test. The `OUT` folder contains all the output `.out` files from the executions, organized in subfolders. These we will often name upper-case letters, and we will refer to them a such.

The `README.md`, as it's name suggests, is the explanation of each test. It contains objective, methodology, execution, results and conclusion headings that discuss each part of the test, and take you through all the process.

The script `run.py` is the heart of the test, it contains the main functioning. For each one, it generates the necessary jobs, submits them, waits for their execution, gathers the output, performs calculations on the numbers, and outputs results. All the working have been designed to the necessity of the test, motivated in the first heading of each README file. It often accepts a variety of arguments, each explained in the corresponding README.

The results from the previous script get written into `results.md`. They go under separate headings, which mark the subfolder of the `OUT` directory. It is accumulative, so running the test multiple times will store all results in there.

Each test can be executed from the test's directory using the desired arguments, but we have created another script that lives in the root directory, `run_test.py`. It's only purpose is to accept the test number which it will call, and additional arguments which will all be included in the call.

Using this design, all tests are completely reproducible. Just run `python3 run_test.py 002` and the whole test will be executed. In a couple of seconds (progress shown on standard output), the results will be visible in the corresponding `TEST/TEST_002/results.md` file. A simple look at the Python script will enable the user to perform more complicated and personalized tests (for example varying the number of executions back to back of the compiled code).

## Scheduler

We have created a personalized `scheduler.sub`, that adapts to our needs for the tests.

The scheduler file takes in 6 arguments:

1. test number: the text following the underscore (e.g. "001" in the case of `TEST_001`)
2. flags: all uppercase, separeted by a comma "," (e.g "FEED_INPUT,TRAINING_BACK_PROP_ERRORS")
3. repetitions: number of times that the same compiled executable is ran back to back
4. subfolder: specific subfolder inside `OUT/` to store output files
5. training file: path to the `training.c` file (relevant for `TEST_002`)
6. configuration file: path to the `configfile.txt` file (relevant for `TEST_003`)
6. file path of the `training.c` file (relevant for `TEST_002`)

It always redirects the ouput of the code to files inside each `OUT` folder in each test. Nonetheless, it always leaves a "see [...]" note in the default slurm file pointing to the right location. All the flags are parsed and given to the compiler.

## Output files

Since we have defined a personalized scheduler, we have structured the ouput file in a way that simplifies analysis and improves version and results control. Their name is always `test_{slurm_job_id}.out`.

For each job, the output file contains custom metadata with all the information about the job and other configurations that are necessary for later analysis or debugging. The file's properties are:

- `test_number`: the name of the test (e.g. "001" for `TEST_001`)
- `date`: the date in YYYY-MM-DD HH:mm:ss format
- `job_id`: the slurm job id number
- `output_directory`: the directory it lives in
- `output_file`: the name of the file
- `compilation_flags`: parallelization flags (e.g. "-DALL" or "-DOP")
- `number_executions`: the number of times the executable is ran
- `server`: the alias of the server
- `partition_file`: the used partition file name
- `num_threads`: number of threads used
- `num_epochs`: number of epochs
- `num_neurons`: number of neurons in the hidden layer

After the metadata, each execution is marked with `#START:id` at the start and `#END:id` at the end, where `id` is the index of the execution. This way, if we execute the code multiple times we know exactly where does each output come from. In between these two comments, we have all the program's output.

# Conclusions

TODO