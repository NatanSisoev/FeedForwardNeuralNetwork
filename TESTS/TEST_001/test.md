# TEST_001

## Objective

The primary objective of this initial test is to simply find the optimal combination of parallelizations (which we will call `TAGS`). We will use a brute-force tecnique, simply iterating over the $2^7=128$ permutations and see which one performs better.

We know that not all parallelizations are helpful, some tasks are so short that the creation of multiple threads itself takes longer than the actual tasks. That's the reason why we have to filter out some of the parallelizations we've made.


## Methodology

Fent servir la llibreria `itertools` de `Python` som capaços de generar fàcilment totes les possibles combinacions d'etiquetes. Les executem fent servir l`scheduler` i analitzem els resultats.

## Execution

A `Python` [script](run.py) has been made to carry out this test.

It can be executed from the root folder using [this](../../run_test.py) other script with the following command:

```bash
python3 run_test.py 001 [SUBFOLDER, MODE]
```

Available options are:

- `SUBFOLDER`: subfolder inside `OUT/` to output `.out` files` (the name of the test)
- `MODE`: either `e` to execute the test, `a` to analyze the results, or both

No arguments at all will create a new subfolder named with the first unused upper case letter, where you will find all the `.out` files. Results will be available in [this](results.md) file under the subfolder's header.

## Results

For our biggest test, we executed each combination 100 times, and then calcualte the average (test `A`). In total, the program ran for almost 4 hours (at night). The following ranking was created:

| #   | avg       | std      | min       | max       | n   | rel_std  | range_ratio | flags                                           |
|-----|-----------|----------|-----------|-----------|-----|----------|-------------|------------------------------------------------|
| 001 | 3.473646  | 0.238805 | 3.255061  | 5.605700  | 100 | 0.068748 | 1.722149    | `FI`, `FPL`, `BPE`, `BPO`, `BPH`, `UWW`, `UWB` |
| 002 | 3.484115  | 0.198727 | 3.236850  | 4.643671  | 100 | 0.057038 | 1.434627    | `FPL`, `BPE`, `BPO`, `BPH`, `UWW`, `UWB`       |
| 003 | 3.550909  | 1.146911 | 2.258562  | 8.818422  | 100 | 0.322991 | 3.904441    | `FI`, `FPL`, `BPE`, `BPH`, `UWW`, `UWB`       |
| ... | ...       | ...      | ...       | ...       | ... | ...      | ...         | ...                                            |
| 117 | 12.979604 | 1.141285 | 10.125176 | 15.171276 | 100 | 0.087929 | 1.498372    | `BPH`                                          |
| 118 | 12.985589 | 1.211251 | 10.023879 | 15.756089 | 99  | 0.093277 | 1.571855    | `NONE`                                        |
| 119 | 13.020803 | 1.287519 | 10.779306 | 21.959309 | 100 | 0.098882 | 2.037173    | `FI`, `FPL`, `UWB`                            |
| ... | ...       | ...      | ...       | ...       | ... | ...      | ...         | ...                                            |
| 126 | 13.147614 | 1.254910 | 10.153961 | 15.535125 | 100 | 0.095448 | 1.529957    | `BPO`                                          |
| 127 | 13.269134 | 1.221293 | 8.199807  | 16.571745 | 100 | 0.092040 | 2.020992    | `FPL`                                          |
| 128 | 13.314294 | 1.070659 | 8.201717  | 18.346678 | 100 | 0.080414 | 2.236931    | `FI`, `BPO`, `UWW`                            |

For now, very good results. We see that basically all parallelization out-perfmormed the sequential program (but 10 combinations). The top three have achieved a total speedup of around 3.7, which for now is very good.

Investigating the total results, we find some interesting insights:

| #   | avg (s)   | std     | min       | max       | n   | rel_std  | range_ratio | tags                                         |
|-----|-----------|---------|-----------|-----------|-----|----------|-------------|----------------------------------------------|
| 039 | 9.410032  | 1.511699| 1.908416  | 13.167567 | 100 | 0.160648 | 6.899736    | `FPL`, `BPO`, `UWW`                         |
| 002 | 3.484115  | 0.198727| 3.236850  | 4.643671  | 100 | 0.057038 | 1.434627    | `FPL`, `BPE`, `BPO`, `BPH`, `UWW`, `UWB`   |
| 020 | 8.253980  | 2.844047| 2.177416  | 15.607096 | 100 | 0.344567 | 7.167714    | `FI`, `FPL`, `BPH`, `UWW`, `UWB`           |
| 006 | 3.683354  | 1.566816| 2.330171  | 11.681657 | 100 | 0.425377 | 5.013219    | `FI`, `FPL`, `BPE`, `BPO`, `BPH`, `UWW`    |
| 067 | 11.063307 | 2.784355| 2.189258  | 20.044112 | 100 | 0.251675 | 9.155665    | `FPL`, `BPH`, `UWW`, `UWB`                 |

In the order of the rows, we see:

- minimum achieved time: 1.908416

    > over-all lowest time

- minimum standard deviation: 0.198727

    > at best, the server gives a standard variation of 0.2 seconds

- maximum standard deviation: 2.844047

    > at worst, the server gives a standard variation of 2.8 seconds

- maximum relative standard deviation: 0.425377

    > at worst, the server gives more than 40% difference to the average one out of three times

- maximum range ratio: 9.155665

    > at worst, the server gives times almost 10 times bigger than others

All these statistics are given over 100 individual runs, so they have pretty big significance (statistical confidence). More than anything, this goes to show that the server is far too unreliable to test small improvements that don't exceed at least 10% of the total time. This means that, with the current configurations of the neural network, any micro-optimization that scrapes off .05 seconds of the total time is not testable.

For the sake of the server, let's run the script again (only 1 repetition) and see the results (test `B`):

| #   | avg      | std      | min      | max      | n   | rel_std  | range_ratio | flags                                   |
|-----|----------|----------|----------|----------|-----|----------|-------------|----------------------------------------|
| 001 | 2.084486 | 0.000000 | 2.084486 | 2.084486 | 1   | 0.000000 | 1.000000    | `FPL`, `BPH`, `UWW`                    |
| 002 | 2.134077 | 0.000000 | 2.134077 | 2.134077 | 1   | 0.000000 | 1.000000    | `FPL`, `BPE`, `BPH`, `UWW`             |
| 003 | 2.162706 | 0.000000 | 2.162706 | 2.162706 | 1   | 0.000000 | 1.000000    | `FI`, `FPL`, `BPE`, `BPH`, `UWW`, `UWB`|
| ... | ...      | ...      | ...      | ...      | ... | ...      | ...         | ...                                    |
| 001 | 1.968112 | 0.000000 | 1.968112 | 1.968112 | 1   | 0.000000 | 1.000000    | `FPL`, `BPH`, `UWB`                    |
| 002 | 2.032400 | 0.000000 | 2.032400 | 2.032400 | 1   | 0.000000 | 1.000000    | `FPL`, `BPH`, `UWW`                    |
| 003 | 2.147710 | 0.000000 | 2.147710 | 2.147710 | 1   | 0.000000 | 1.000000    | `FI`, `FPL`, `BPE`, `BPH`, `UWW`      |
| ... | ...      | ...      | ...      | ...      | ... | ...      | ...         | ...                                    |
| 001 | 2.054270 | 0.000000 | 2.054270 | 2.054270 | 1   | 0.000000 | 1.000000    | `FPL`, `BPE`, `BPH`, `UWW`             |
| 002 | 2.097778 | 0.000000 | 2.097778 | 2.097778 | 1   | 0.000000 | 1.000000    | `FI`, `FPL`, `BPH`, `UWW`              |
| 003 | 2.100028 | 0.000000 | 2.100028 | 2.100028 | 1   | 0.000000 | 1.000000    | `FPL`, `BPH`, `UWW`, `UWB`             |

We see that the times we now get are much better than before, going down to 1.97 in one case. Again, it is very clear that the server varaibility plays a very big role in this program.

Nonetheless, notice the frequencies of the tags:

- `FPL`: 9 times
- `BPH`: 9 times
- `UWW`: 8 times
- `BPE`: 3 times
- `UWB`: 2 times
- `FI`: 2 times

We have three clear winners for now: `TRAINING_FORWARD_PROP_LAYERS`, `TRAINING_BACK_PROP_HIDDEN_LAYERS` and `TRAINING_UPDATE_WEIGHTS_WEIGHTS`.

In the following tests we will see if these conclusions are really backed up by the individual speedup of each parallelization.

## Conclusions

- The main takeaway from this test is that the server is very unpredictable.
- Various runs of the same code may vary up to more than a third of the total time.
- We see from the last three individual executions that 3 of the tags consistently show up in the best performing combinations.
- We have pretty solid conclusions from the first test, which has a high confidence because of the large number of repetitions.
- We will solidify these conclusions in the following tests.
