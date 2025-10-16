# RESULTS (2025-10-16 20:35:31)
sequential (1 run)
```
Processed 1 files, average total elapsed = 10.031208 s

Phase breakdown (average per file):
forward_propagation      :  4.274663 s ( 42.61%)
backward_propagation     :  2.732204 s ( 27.24%)
weight_update            :  2.668826 s ( 26.61%)
predicting               :  0.208829 s (  2.08%)
loading_patterns         :  0.042433 s (  0.42%)
input_feed               :  0.015177 s (  0.15%)
pattern_shuffle          :  0.000499 s (  0.00%)
recognition              :  0.000095 s (  0.00%)
```

# RESULTS (2025-10-16 20:36:39)
sequential (10 runs)
```
Processed 10 files, average total elapsed = 10.025178 s

Phase breakdown (average per file):
forward_propagation      :  4.283110 s ( 42.72%)
weight_update            :  2.684651 s ( 26.78%)
backward_propagation     :  2.654150 s ( 26.47%)
predicting               :  0.210375 s (  2.10%)
loading_patterns         :  0.056757 s (  0.57%)
input_feed               :  0.015724 s (  0.16%)
pattern_shuffle          :  0.000503 s (  0.01%)
recognition              :  0.000095 s (  0.00%)
```

# RESULTS (2025-10-16 20:39:24)
parallel `OPT` (10 runs)
```
Processed 10 files, average total elapsed = 2.030740 s

Phase breakdown (average per file):
forward_propagation      :  1.034072 s ( 50.92%)
backward_propagation     :  0.418917 s ( 20.63%)
weight_update            :  0.386745 s ( 19.04%)
predicting               :  0.051131 s (  2.52%)
loading_patterns         :  0.039584 s (  1.95%)
input_feed               :  0.032872 s (  1.62%)
pattern_shuffle          :  0.000502 s (  0.02%)
recognition              :  0.000153 s (  0.01%)
```

# RESULTS (2025-10-16 21:05:46)
sequential (20 runs)
```
Processed 20 files, average total elapsed = 10.030956 s

Phase breakdown (average per file):
forward_propagation      :  4.283941 s ( 42.71%)
weight_update            :  2.686664 s ( 26.78%)
backward_propagation     :  2.658612 s ( 26.50%)
predicting               :  0.209569 s (  2.09%)
loading_patterns         :  0.056804 s (  0.57%)
input_feed               :  0.015834 s (  0.16%)
pattern_shuffle          :  0.000505 s (  0.01%)
recognition              :  0.000095 s (  0.00%)
```

# RESULTS (2025-10-16 21:06:13)
parallel `OPT` (20 runs)
```
Processed 20 files, average total elapsed = 2.046550 s

Phase breakdown (average per file):
forward_propagation      :  1.040508 s ( 50.84%)
backward_propagation     :  0.423518 s ( 20.69%)
weight_update            :  0.394067 s ( 19.26%)
predicting               :  0.050894 s (  2.49%)
loading_patterns         :  0.037077 s (  1.81%)
input_feed               :  0.033087 s (  1.62%)
pattern_shuffle          :  0.000501 s (  0.02%)
recognition              :  0.000154 s (  0.01%)
```
