# feed_input
## sequential
2,357473123 seconds time elapsed
2,495234572 seconds time elapsed
2,303820162 seconds time elapsed
2,370526786 seconds time elapsed
2,300920642 seconds time elapsed

2.357473123 + 2.495234572 + 2.303820162 + 2.370526786 + 2.300920642

    = 11.827975285

11.827975285 / 5 = 2.365595057

## parallel
2,445114370 seconds time elapsed
2,459992216 seconds time elapsed
2,458524924 seconds time elapsed
2,663865484 seconds time elapsed
2,611471927 seconds time elapsed

2.445114370 + 2.459992216 + 2.458524924 + 2.663865484 + 2.611471927

    = 12.638968921

12.638968921 / 5 = 2.5277937842

# forward_prop_j

## sequential
## parallel

---

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

best so far:
TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS UPDATE_WEIGHTS
= 2.121777 s

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

> testing all possible combinations of functions

[RESULTS](TESTS/TEST_002/results.md)

```
Best tag combination: FORWARD_PROP BACK_PROP UPDATE_WEIGHTS
Smallest time: 3.142874
```

`ERROR: Value conversion error in file: TESTS/TEST_002/OUT/TEST_002.sub_63161.out`: still error for tags: 'UPDATE_WEIGHTS'
