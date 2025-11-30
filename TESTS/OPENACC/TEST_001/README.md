 
================================================================================
ANÀLISI DE RESULTATS OPENACC
================================================================================

S'han llegit 128 tests correctament.


================================================================================
ANÀLISI DE VARIÀNCIA
================================================================================

--- Temps d'execució ---
Mitjana global: 9.792504 segons
Desviació estàndard: 2.603290 segons
Variància: 6.777119
Rang: [6.481270, 13.568936]
Desviació estàndard mitjana dels tests: 0.070698

--- Accuracy ---
Mitjana global: 99.89
Desviació estàndard: 33.18
Variància: 1100.74
Rang: [80, 300]
Desviació estàndard mitjana dels tests: 0.01

--- Coeficient de variació ---
Temps: 26.58%
Accuracy: 33.21%

================================================================================
TOP 5 TESTS MÉS RÀPIDS
================================================================================

1. Test #001
   Temps: 6.481270s (±0.071200)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,

2. Test #002
   Temps: 7.154617s (±0.069801)
   Accuracy: 104.00 (±0.00)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_UPDATE_WEIGHTS_BIASES

3. Test #003
   Temps: 7.156712s (±0.061176)
   Accuracy: 104.00 (±0.00)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS

4. Test #004
   Temps: 7.157271s (±0.076361)
   Accuracy: 104.00 (±0.00)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_UPDATE_WEIGHTS_WEIGHTS

5. Test #005
   Temps: 7.158768s (±0.076331)
   Accuracy: 104.00 (±0.00)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS

--- Comparació Accuracy ---
Accuracy mitjana: 100.60
Millor accuracy: 104.00
Pitjor accuracy: 87.00
Rang: 17.00

================================================================================
TOP 5 TESTS MÉS LENTS
================================================================================

1. Test #128
   Temps: 13.568936s (±0.064415)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,FEED_INPUT TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

2. Test #127
   Temps: 13.566904s (±0.077059)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,FEED_INPUT TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

3. Test #126
   Temps: 13.411449s (±0.072160)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,FEED_INPUT TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_UPDATE_WEIGHTS_BIASES

4. Test #125
   Temps: 13.385178s (±0.062172)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,FEED_INPUT TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_BIASES

5. Test #124
   Temps: 13.382727s (±0.070970)
   Accuracy: 87.00 (±0.00)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_OUTPUT_LAYER

--- Comparació Accuracy ---
Accuracy mitjana: 87.00
Millor accuracy: 87.00
Pitjor accuracy: 87.00
Rang: 0.00

================================================================================
TOP 5 TESTS AMB MILLOR ACCURACY
================================================================================

1. Test #042
   Accuracy: 300.50 (±0.71)
   Temps: 7.625256s (±0.105440)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_BIASES

2. Test #043
   Accuracy: 300.00 (±0.00)
   Temps: 7.627962s (±0.079262)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

3. Test #045
   Accuracy: 300.00 (±0.00)
   Temps: 7.647927s (±0.072014)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

4. Test #036
   Accuracy: 130.00 (±0.00)
   Temps: 7.558346s (±0.070194)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_BIASES

5. Test #037
   Accuracy: 130.00 (±0.00)
   Temps: 7.564538s (±0.085454)
   Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

--- Comparació Temps ---
Temps mitjà: 7.604806s
Més ràpid: 7.558346s
Més lent: 7.647927s
Rang: 0.089581s

================================================================================
TOP 5 TESTS AMB PITJOR ACCURACY
================================================================================

1. Test #017
   Accuracy: 80.00 (±0.00)
   Temps: 7.343419s (±0.042420)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

2. Test #020
   Accuracy: 80.00 (±0.00)
   Temps: 7.382687s (±0.032889)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

3. Test #024
   Accuracy: 80.00 (±0.00)
   Temps: 7.418582s (±0.065973)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_BIASES

4. Test #025
   Accuracy: 80.00 (±0.00)
   Temps: 7.420999s (±0.062517)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_WEIGHTS

5. Test #026
   Accuracy: 80.00 (±0.00)
   Temps: 7.422980s (±0.084359)
   Config: OPENACC,TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_UPDATE_WEIGHTS_WEIGHTS TRAINING_UPDATE_WEIGHTS_BIASES

--- Comparació Temps ---
Temps mitjà: 7.397733s
Més ràpid: 7.343419s
Més lent: 7.422980s
Rang: 0.079561s

================================================================================
CONCLUSIONS
================================================================================

--- Test òptim (millor balance temps/accuracy) ---
Test #042
Temps: 7.625256s
Accuracy: 300.50
Config: OPENACC,FEED_INPUT TRAINING_FORWARD_PROP_LAYERS TRAINING_BACK_PROP_ERRORS TRAINING_BACK_PROP_OUTPUT_LAYER TRAINING_BACK_PROP_HIDDEN_LAYERS TRAINING_UPDATE_WEIGHTS_BIASES
Score: 0.9193

--- Correlació Temps-Accuracy ---
Coeficient de correlació: -0.1175
Correlació dèbil: El temps i l'accuracy són independents.

--- Resum estadístic ---
Total de tests analitzats: 128
Rang de temps: 6.481270s - 13.568936s (diferència: 7.087666s)
Rang d'accuracy: 80.00 - 300.50 (diferència: 220.50)

================================================================================
ANÀLISI COMPLETADA
================================================================================