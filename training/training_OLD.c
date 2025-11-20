// sourcecode_delivery (well, modified by sir natan, aka pelu)

/*
 *  training.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 31 gen. 2019
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions per entrenar la xarxa neuronal.
 *
 */

#include "training.h"

#include <math.h>

#include <stdio.h>

/**
 * @brief Iniciatlitza la capa incial de la xarxa (input layer) amb l'entrada
 * que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
    #if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT)
    #pragma omp parallel for  // training.feed_input
    #endif
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];
}

/**
 * @brief Propagació dels valors de les neurones de l'entrada (valors a la input
 * layer) a la resta de capes de la xarxa fins a obtenir una predicció (sortida)
 *
 * @details La capa d'entrada (input layer = capa 0) ja ha estat inicialitzada
 * amb els valors de l'entrada que volem reconeixer. Així, el for més extern
 * (sobre i) recorre totes les capes de la xarxa a partir de la primera capa
 * hidden (capa 1). El for intern (sobre j) recorre les neurones de la capa i
 * calculant el seu valor d'activació [lay[i].actv[j]]. El valor d'activació de
 * cada neurona depén de l'exitació de la neurona calculada en el for més intern
 * (sobre k) [lay[i].z[j]]. El valor d'exitació s'inicialitza amb el biax de la
 * neurona corresponent [j] (lay[i].bias[j]) i es calcula multiplicant el valor
 * d'activació de les neurones de la capa anterior (i-1) pels pesos de
 * les connexions (out_weights) entre les dues capes. Finalment, el valor
 * d'activació de la neurona (j) es calcula fent servir la funció RELU
 * (REctified Linear Unit) si la capa (j) és una capa oculta (hidden) o la
 * funció Sigmoid si es tracte de la capa de sortida.
 *
 */
void forward_prop() {
    for (int i = 1; i < num_layers; i++) {
        #if defined(ALL) || defined(TRAINING) || defined(FORWARD_PROP) || defined(TRAINING_FORWARD_PROP_LAYERS) || defined(OPT)
        #if defined(OPENMP)
        #pragma omp parallel for  // training.forward_prop.layers
        #elif defined(OPENACC)
        #pragma acc parallel loop  //present(lay[0:num_layers], num_neurons[0:num_layers]) gang vector  // training.forward_prop.layers
        #endif
        #endif
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].z[j] = lay[i].bias[j];
            for (int k = 0; k < num_neurons[i - 1]; k++)
                lay[i].z[j] += ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) * (lay[i - 1].actv[k]));

            if (i < num_layers - 1)
                lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
            else
                lay[i].actv[j] = 1 / (1 + exp(-lay[i].z[j]));
        }
    }
}

/**
 * @brief Calcula el gradient que es necessari aplicar als pesos de les
 * connexions entre neurones per corregir els errors de predicció
 *
 * @details Calcula dos vectors de correcció per cada capa de la xarxa, un per
 * corregir els pesos de les connexions de la neurona (j) amb la capa anterior
 *          (lay[i-1].dw[j]) i un segon per corregir el biax de cada neurona de
 * la capa actual (lay[i].bias[j]). Hi ha un tractament diferent per la capa de
 * sortida (num_layesr -1) perquè aquest és l'única cas en el que l'error es
 * conegut (lay[num_layers-1].actv[j] - desired_outputs[p][j]). Això es pot
 * veure en els dos primers fors. Per totes les capes ocultes (hidden layers) no
 * es pot saber el valor d'activació esperat per a cada neurona i per tant es fa
 * una estimació. Aquest càlcul es fa en el doble for que recorre totes les
 * capes ocultes (sobre i) neurona a neurona (sobre j). Es pot veure com en cada
 * cas es fa una estimació de quines haurien de ser les activacions de les
 * neurones de la capa anterior (lay[i-1].dactv[k] = lay[i-1].out_weights[j*
 * num_neurons[i-1] + k] * lay[i].dz[j];), excepte pel cas de la capa d'entrada
 * (input layer) que és coneguda (imatge d'entrada).
 *
 */
void back_prop(int p) {
    // Output Layer
    /*
    Càlcul de l'error local (δ_j) a la capa de sortida
    Representa com la pre-activació z_j (abans de la funció d'activació)
    contribueix a l'error quadràtic total E = 0.5 * Σ (a_j - y_j)^2
    Per a la funció d'activació sigmoide, 
    # f(z_j) = 1 / (1 + exp(-z_j)) = a_j
    i la seva derivada és
    # f'(z_j) = a_j * (1 - a_j)
    # δ_j = ∂E/∂z_j =  ∂E/∂a_j * ∂a_j/∂z_j = (a_j - y_j) * f'(z_j)
    ___
    Càlcul del gradient del biax (∂E/∂b_j) a la capa de sortida
    Tenim: z_j = b_j + Σ w_jk * a_k
    Per tant, el gradient del biax és igual a l'error local
    # ∂E/∂b_j =  ∂E/∂z_j * ∂z_j/∂b_j = ∂E/∂z_j = δ_j
    */
    #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_ERRORS)
    #pragma omp parallel for  // training.back_prop.errors
    #endif
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        lay[num_layers - 1].dz[j] =
            (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
            (lay[num_layers - 1].actv[j]) * (1 - lay[num_layers - 1].actv[j]);
        lay[num_layers - 1].dbias[j] = lay[num_layers - 1].dz[j];
    }

    // Calcula els gradients dels pesos de l'última capa oculta cap a la capa de sortida:
    //   dw_kj[] = actv[k] * dz[j], és a dir, com ajustar cada pes segons l'error de la sortida.
    // Propaga l'error cap enrere a la capa oculta:
    //   dactv[k] = w_kj[] * dz[j],
    // que després servirà per calcular dz de les neurones ocultes
    
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_OUTPUT_LAYER)
        #pragma omp parallel for  // training.back_prop.output_layer
        #endif
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] = (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
            lay[num_layers - 2].dactv[k] = lay[num_layers - 2].out_weights[j * num_neurons[num_layers - 2] + k] * lay[num_layers - 1].dz[j];
        }
    }
    
    // Hidden Layers
    for (int i = num_layers - 2; i > 0; i--) {
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_HIDDEN_LAYERS) || defined(OPT)
        #if defined(OPENMP)
        #pragma omp parallel for  // training.back_prop.hidden_layers
        #elif defined(OPENACC)
        #pragma acc parallel loop  //present(lay[0:num_layers], num_neurons[0:num_layers]) gang vector  // training.back_prop.hidden_layers
        #endif
        #endif
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0;

            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i - 1].dw[j * num_neurons[i - 1] + k] = lay[i].dz[j] * lay[i - 1].actv[k];
                if (i > 1) {
                    #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_HIDDEN_LAYERS) || defined(OPT)
                    #if defined(OPENMP)
                    #pragma omp critical
                    #elif defined(OPENACC)
                    #pragma acc atomic update
                    #endif
                    #endif
                    lay[i - 1].dactv[k] += lay[i - 1].out_weights[j * num_neurons[i - 1] + k] * lay[i].dz[j];
                }
            }

            lay[i].dbias[j] = lay[i].dz[j];
        }
    }
}

/**
 * @brief Actualitza els vectors de pesos (out_weights) i de biax (bias) de cada
 * etapa d'acord amb els càlculs fet a la funció de back_prop i el factor
 * d'aprenentatge alpha
 *
 * @see back_prop
 */
void update_weights(void) {
    for (int i = 0; i < num_layers - 1; i++) {
        #if defined(ALL) || defined(TRAINING) || defined(UPDATE_WEIGHTS) || defined(TRAINING_UPDATE_WEIGHTS_WEIGHTS) || defined(OPT)
        #if defined(OPENMP)
        #pragma omp parallel for  // training.update_weights.weights
        #elif defined(OPENACC)
        #pragma acc parallel loop collapse(2)  //present(lay[0:num_layers], num_neurons[0:num_layers]) gang vector  // training.update_weights.weights
        #endif
        #endif
        for (int j = 0; j < num_neurons[i + 1]; j++)
            for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                lay[i].out_weights[j * num_neurons[i] + k] = lay[i].out_weights[j * num_neurons[i] + k] - alpha * lay[i].dw[j * num_neurons[i] + k];

        #if defined(ALL) || defined(TRAINING) || defined(UPDATE_WEIGHTS) || defined(TRAINING_UPDATE_WEIGHTS_BIASES)
        #if defined(OPENMP)
        #pragma omp parallel for  // training.update_weights.biases
        #elif defined(OPENACC)
        #pragma acc parallel loop  //present(lay[0:num_layers], num_neurons[0:num_layers]) gang vector  // training.update_weights.biases
        #endif
        #endif
        for (int j = 0; j < num_neurons[i]; j++)  // Update Bias
            lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);
    }
}


/**
 *  main.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de
 *  l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona Created on: 31
 *  gen. 2019 Last modified: fall 24 (curs 24-25) Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funció que entrena la xarxa neuronal definida + Funció que fa el test del
 *  model entrenat + programa principal.
 *
 */

#include "main.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

//-----------FREE INPUT------------
void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized(int p, layer Output) {
    int imax = 0;

    for (int i = 1; i < num_out_layer; i++)
        if (Output.actv[i] > Output.actv[imax])
            imax = i;

    if (imax == Validation[p])
        total++;

    if (debug == 1) {
        printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax,
               Validation[p]);
        for (int k = 0; k < num_out_layer; k++)
            printf("\t%f\t", Output.actv[k]);
        printf("\n");
    }
}

/**
 * @brief Entrena la xarxa neuronal en base al conjunt d'entrenament
 *
 * @details Primer carrega tots els patrons d'entrenament (loadPatternSet)
 *          Després realitza num_epochs iteracions d'entrenament.
 *          Cada epoch fa:
 *              - Determina aleatòriament l'ordre en que es consideraran els
 * patrons (per evitar overfitting)
 *              - Per cada patró d'entrenament fa el forward_prop (reconeixament
 * del patró pel model actual) i el back_prop i update_weights (ajustament de
 * pesos i biaxos per provar de millorar la precisió del model)
 *
 * @see loadPatternSet, feed_input, forward_prop, back_prop, update_weights,
 * freeInput
 *
 */
void train_neural_net() {
    // printf("\nTraining...\n");

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    #if defined(OPENACC_COPY)
    #pragma acc enter data copyin(input[0:num_training_patterns])
    for (int r = 0; r < num_training_patterns; r++) {
        int n = num_neurons[0];
        #pragma acc enter data copyin(input[r][0:n])
    }
    #endif

    int ranpat[num_training_patterns];

    // Gradient Descent
    for (int it = 0; it < num_epochs; it++) {
        // Train patterns randomly
        for (int p = 0; p < num_training_patterns; p++)
            ranpat[p] = p;

        for (int p = 0; p < num_training_patterns; p++) {
            int x = rando();
            int np = (x * x) % num_training_patterns;
            int op = ranpat[p];
            ranpat[p] = ranpat[np];
            ranpat[np] = op;
        }

        for (int i = 0; i < num_training_patterns; i++) {
            int p = ranpat[i];

            feed_input(p);
            #if defined(OPENACC_COPY)
            int n = num_neurons[0];
            #pragma acc update device(lay[0].actv[0:n])
            #endif
            forward_prop();
            back_prop(p);
            update_weights();
        }
    }

    #if defined(OPENACC_COPY)
    for (int r = 0; r < num_training_patterns; r++) {
        int n = num_neurons[0];
        #pragma acc exit data delete(input[r][0:n])
    }
    #pragma acc exit data delete(input[0:num_training_patterns])
    #endif

    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    char** rSet;

    // printf("\nTesting...\n");

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        exit(-1);
    }

    #if defined(OPENACC_COPY)
    #pragma acc enter data copyin(rSet[0:num_test_patterns])
    for (int r = 0; r < num_test_patterns; r++) {
        int n = num_neurons[0];
        #pragma acc enter data copyin(rSet[r][0:n])
    }
    #endif

    for (int i = 0; i < num_test_patterns; i++) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];

        #if defined(OPENACC_COPY)
        int n = num_neurons[0];
        #pragma acc update device(lay[0].actv[0:n])
        #endif
        
        forward_prop();

        #if defined(OPENACC_COPY)
        #pragma acc update host(lay[num_layers - 1].actv[0:num_neurons[num_layers - 1]])
        #endif

        printRecognized(i, lay[num_layers - 1]);
    }

    // printf("\nTotal encerts = %d\n", total);
    printf("%d\t", total);

    #if defined(OPENACC_COPY)
    for (int r = 0; r < num_test_patterns; r++) {
        #pragma acc exit data delete(rSet[r][0:num_neurons[0]])
    }
    #pragma acc exit data delete(rSet[0:num_test_patterns])
    #endif

    freeInput(num_test_patterns, rSet);
}

//-----------MAIN-----------//
int main(int argc, char** argv) {
    if (debug == 1)
        printf("argc = %d \n", argc);
    if (argc <= 1)
        readConfiguration("configuration/configfile.txt");
    else
        readConfiguration(argv[1]);

    if (debug == 1)
        printf("FINISH CONFIG \n");

    // Initialize the neural network module
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    if (debug == 1)
        printf("COST MALLOC \n");

    cost = (float*)malloc(num_neurons[num_layers - 1] * sizeof(float));

    #if defined(OPENACC_COPY)
    #pragma acc enter data copyin(num_layers)
    #pragma acc enter data copyin(num_neurons[0:num_layers])
    #pragma acc enter data copyin(lay[0:num_layers])

    for (int i = 0; i < num_layers; i++) {
        #pragma acc enter data copyin(lay[i].actv[0:num_neurons[i]])
        #pragma acc enter data copyin(lay[i].z[0:num_neurons[i]])
        #pragma acc enter data copyin(lay[i].bias[0:num_neurons[i]])

        if (i < num_layers - 1) {
            long ow_size = (long)num_neurons[i + 1] * (long)num_neurons[i];
            #pragma acc enter data copyin(lay[i].out_weights[0:ow_size])
            #pragma acc enter data create(lay[i].dw[0:ow_size])
            #pragma acc enter data create(lay[i].dbias[0:num_neurons[i + 1]])
        }

        #pragma acc enter data create(lay[i].dactv[0:num_neurons[i]])
    }
    #endif

    // Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Train
    train_neural_net();

    // Test
    test_nn();

    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    free(cost);

    // printf("\n\nGoodbye! (%f sec)\n\n", elapsed);
    printf("%f\n", elapsed);

    return 0;
}



/*
#pragma acc enter data copyin()
alpha
num_out_layer
num_layers
num_neurons[0:num_layers]
desired_outputs[0:num_out_layer]
input[0:num_training_patterns]

for1
input[i][0:num_neurons[0]]
desired_outputs[0:num_out_layer]
lay[0:num_layers]

for2
lay[i].actv[0:num_neurons[i]]
lay[i].bias[0:num_neurons[i]]
lay[i].z[0:num_neurons[i]]
lay[i].dactv[0:num_neurons[i]]
lay[i].dbias[0:num_neurons[i]]
lay[i].dz[0:num_neurons[i]]

for3
lay[i].out_weights[0:(num_neurons[i+1]*num_neurons[i])]
lay[i].dw[0:(num_neurons[i+1]*num_neurons[i])

*/