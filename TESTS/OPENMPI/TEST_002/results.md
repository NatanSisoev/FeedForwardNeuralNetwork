# TEST_002 Results - Subfolder I

*Generated from 14 output files*

## Experiment 2.1: Strong Scaling (Processos)

*Configuration: 10 epochs, 135 neurons, 1 node*

| Tasks | Train (s) | Test (s) | Total (s) | Speedup | Efficiency (%) |
|-------|-----------|----------|-----------|---------|----------------|
|     2 |    5.1399 |   0.1137 |    5.2539 |    1.00 |           50.0 |
|     4 |    2.5805 |   0.0614 |    2.6422 |    1.99 |           49.7 |
|     6 |    1.7638 |   0.0459 |    1.8102 |    2.90 |           48.4 |
|     8 |    1.3935 |   0.0359 |    1.4299 |    3.67 |           45.9 |

## Experiment 2.2: Escalabilitat per Èpoques

*Configuration: 8 tasks, 135 neurons, 1 node*

| Epochs | Train (s) | Test (s) | Total (s) | Accuracy |
|--------|-----------|----------|-----------|----------|
|      1 |    0.1647 |   0.0360 |    0.2012 |      849 |
|     10 |    1.3935 |   0.0359 |    1.4299 |      914 |
|     50 |    6.6950 |   0.0442 |    6.7396 |      921 |
|    100 |   13.4133 |   0.0418 |   13.4558 |      923 |
|    200 |   26.7718 |   0.0466 |   26.8189 |      925 |

## Experiment 2.3: Escalabilitat per Neurones

*Configuration: 8 tasks, 10 epochs, 1 node*

| Neurons | Train (s) | Test (s) | Total (s) | Accuracy |
|---------|-----------|----------|-----------|----------|
|     135 |    1.3935 |   0.0359 |    1.4299 |      914 |
|     250 |    6.0668 |   0.0594 |    6.1267 |      915 |

## Experiment 2.4: Weak Scaling (Nodes)

*Configuration: 8 tasks per node, 10 epochs, 135 neurons*

| Nodes | Total Tasks | Train (s) | Test (s) | Total (s) | Speedup |
|-------|-------------|-----------|----------|-----------|----------|
|     1 |           8 |    1.3935 |   0.0359 |    1.4299 |     1.00 |
|     4 |          32 |    0.3984 |   0.0164 |    0.4152 |     3.44 |
|     8 |          64 |    0.4573 |   0.0238 |    0.4816 |     2.97 |

---

## Summary

- Total configurations analyzed: 14
- Epochs tested: [1, 10, 50, 100, 200]
- Tasks tested: [2, 4, 6, 8, 32, 64]
- Neurons tested: [135, 250]
- Nodes tested: [1, 4, 8]
