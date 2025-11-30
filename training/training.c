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

#include <sys/time.h>

elapsed_feed_input = 0;
elapsed_forward_prop = 0;
elapsed_back_prop = 0;
elapsed_update_weights = 0;

/**
 * @brief Iniciatlitza la capa incial de la xarxa (input layer) amb l'entrada
 * que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // training.feed_input
    #if defined(OPENMP)
        #if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT)
            #pragma omp parallel for
        #endif
    #elif defined(OPENACC)
        #if defined(ALL) || defined(TRAINING) || defined(FEED_INPUT) || defined(OPT)
            #pragma acc parallel loop async present(lay, input, num_neurons)
        #else
            #pragma acc parallel loop async present(lay, input, num_neurons) seq
        #endif
    #endif
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j];

    #if defined(OPENACC)
        #pragma acc wait
    #endif
    
    gettimeofday(&end, 0);
    elapsed_feed_input += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
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
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    #if defined(OPENACC)
        #pragma acc data present(lay[0:num_layers], num_neurons[0:num_layers])
        {
    #endif
    for (int i = 1; i < num_layers; i++) {
        // training.forward_prop.layers
        #if defined(OPENMP)
            #if defined(ALL) || defined(TRAINING) || defined(FORWARD_PROP) || defined(TRAINING_FORWARD_PROP_LAYERS) || defined(OPT)
                #pragma omp parallel for
            #endif
        #elif defined(OPENACC)
            #if defined(ALL) || defined(TRAINING) || defined(FORWARD_PROP) || defined(TRAINING_FORWARD_PROP_LAYERS) || defined(OPT)
                #pragma acc parallel loop
            #else
                #pragma acc parallel loop seq
            #endif
        #endif
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].z[j] = lay[i].bias[j];
            float temp = lay[i].z[j]; // Temporary variable for reduction
            // training.forward_prop.layers
            #if defined(OPENMP)
                #if defined(ALL) || defined(TRAINING) || defined(FORWARD_PROP) || defined(TRAINING_FORWARD_PROP_LAYERS) || defined(OPT)
                    #pragma omp reduction(+:sum)
                #endif
            #elif defined(OPENACC)
                #if defined(ALL) || defined(TRAINING) || defined(FORWARD_PROP) || defined(TRAINING_FORWARD_PROP_LAYERS) || defined(OPT)
                    #pragma acc loop vector reduction(+ : temp)
                #else
                    #pragma acc loop vector reduction(+ : temp) seq
                #endif
            #endif
            for (int k = 0; k < num_neurons[i - 1]; k++)
                temp += ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) * (lay[i - 1].actv[k]));
            
            lay[i].z[j] = temp;     // Store the result back to the structure  
            if (i < num_layers - 1)
                lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
            else
                lay[i].actv[j] = 1.0 / (1.0 + exp(-lay[i].z[j]));
        }
    }
    #if defined(OPENACC)
        }
    #endif

    gettimeofday(&end, 0);
    elapsed_forward_prop += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
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
 *
== Output Layer
 
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
void back_prop(int p) {
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // TRAINING_BACK_PROP_ERRORS
    #if defined(OPENACC)
        #pragma acc data present(lay[0:num_layers], num_neurons[0:num_layers], desired_outputs[0:10])
        {  
    #endif
    #if defined(OPENMP)
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_ERRORS)
            #pragma omp parallel for
        #endif
    #elif defined(OPENACC)
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_ERRORS) || defined(OPT)
            #pragma acc parallel loop async
        #else
            #pragma acc parallel loop async seq
        #endif
    #endif
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        lay[num_layers - 1].dz[j] =
            (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
            (lay[num_layers - 1].actv[j]) * (1.0 - lay[num_layers - 1].actv[j]);
        lay[num_layers - 1].dbias[j] = lay[num_layers - 1].dz[j];
    }

    // TRAINING_BACK_PROP_OUTPUT_LAYER
    #if defined(OPENACC)
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_OUTPUT_LAYER) || defined(OPT)
            #pragma acc parallel loop async
        #else
            #pragma acc parallel loop async seq
        #endif
    #endif
    for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
        lay[num_layers - 2].dactv[k] = 0.0; // Inicialitzar dactv de l'última capa oculta a zero
    }

    // Calcula els gradients dels pesos de l'última capa oculta cap a la capa de sortida:
    //   dw_kj[] = actv[k] * dz[j], és a dir, com ajustar cada pes segons l'error de la sortida.
    // Propaga l'error cap enrere a la capa oculta:
    //   dactv[k] = w_kj[] * dz[j],
    // que després servirà per calcular dz de les neurones ocultes
    
    /*#if defined(OPENACC)
    #pragma acc parallel loop collapse(2) async
    #endif
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] = 
                lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k];
            
            #if defined(OPENACC)
            #pragma acc atomic update
            #endif
            lay[num_layers - 2].dactv[k] += 
                lay[num_layers - 2].out_weights[j * num_neurons[num_layers - 2] + k] * 
                lay[num_layers - 1].dz[j];
        }
    }*/

    #if defined(OPENACC)
        #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_OUTPUT_LAYER) || defined(OPT)
            #pragma acc parallel loop async
        #else
            #pragma acc parallel loop async seq
        #endif
    #endif
    for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
        float sum = 0;
        #if defined(OPENACC)
            #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_OUTPUT_LAYER) || defined(OPT)
                #pragma acc loop reduction(+:sum)
            #else
                #pragma acc loop reduction(+:sum) seq
            #endif
        #endif
        for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] = 
                lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k];
            
            sum += lay[num_layers - 2].out_weights[j * num_neurons[num_layers - 2] + k] * 
                lay[num_layers - 1].dz[j];
        }

        lay[num_layers - 2].dactv[k] = sum;
    }
    
    // TRAINING_BACK_PROP_HIDDEN_LAYERS
    for (int i = num_layers - 2; i > 0; i--) {
        int Nprev = num_neurons[i - 1];
        // Inicialitzar dactv de la capa anterior si no és la capa d'entrada
        if (i > 1) {
            #if defined(OPENACC)
                #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_HIDDEN_LAYERS) || defined(OPT)
                    #pragma acc parallel loop async
                #else
                    #pragma acc parallel loop async seq
                #endif
            #endif
            for (int k = 0; k < Nprev; k++) {
                lay[i - 1].dactv[k] = 0.0;
            }
        }
        
        #if defined(OPENACC)
            #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_HIDDEN_LAYERS) || defined(OPT)
                #pragma acc parallel loop  // training.back_prop.hidden_layers
            #else
                #pragma acc parallel loop seq
            #endif
        #endif
        for (int j = 0; j < num_neurons[i]; j++) {
            float *restrict w_row = lay[i - 1].out_weights + j * Nprev;
            lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0.0;
            #if defined(OPENACC)
                #if defined(ALL) || defined(TRAINING) || defined(BACK_PROP) || defined(TRAINING_BACK_PROP_HIDDEN_LAYERS) || defined(OPT)
                    #pragma acc loop
                #else
                    #pragma acc loop seq
                #endif
            #endif
            for (int k = 0; k < Nprev; k++) {
                lay[i - 1].dw[j * Nprev + k] = lay[i].dz[j] * lay[i - 1].actv[k];
                if (i > 1) {
                    #if defined(OPENACC)
                            #pragma acc atomic update
                    #endif
                    // TODO: reduction
                    lay[i - 1].dactv[k] += w_row[k] * lay[i].dz[j];
                }
            }

            lay[i].dbias[j] = lay[i].dz[j];
        }
    }
    #if defined(OPENACC)
        }
    #endif

    gettimeofday(&end, 0);
    elapsed_back_prop += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
}

/**
 * @brief Actualitza els vectors de pesos (out_weights) i de biax (bias) de cada
 * etapa d'acord amb els càlculs fet a la funció de back_prop i el factor
 * d'aprenentatge alpha
 *
 * @see back_prop
 */
void update_weights(void) {
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    #if defined(OPENACC)
        #pragma acc data present(lay[0:num_layers], num_neurons[0:num_layers], alpha)
        {
    #endif
    for (int i = 0; i < num_layers - 1; i++) {
        #if defined(OPENACC)
            #if defined(ALL) || defined(TRAINING) || defined(UPDATE_WEIGHTS) || defined(OPT)
                #pragma acc parallel loop collapse(2) async  // training.update_weights.weights
            #else
                #pragma acc parallel loop collapse(2) async seq
            #endif
        #endif
        for (int j = 0; j < num_neurons[i + 1]; j++)
            for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                lay[i].out_weights[j * num_neurons[i] + k] = lay[i].out_weights[j * num_neurons[i] + k] - alpha * lay[i].dw[j * num_neurons[i] + k];

        #if defined(OPENACC)
            #if defined(ALL) || defined(TRAINING) || defined(UPDATE_WEIGHTS) || defined(TRAINING_UPDATE_WEIGHTS_BIASES) || defined(OPT)
                #pragma acc parallel loop async  // training.update_weights.biases
            #else
                #pragma acc parallel loop async seq
            #endif
        #endif
        for (int j = 0; j < num_neurons[i + 1]; j++)  // Update Bias
            lay[i + 1].bias[j] = lay[i + 1].bias[j] - (alpha * lay[i + 1].dbias[j]);
    }
    #if defined(OPENACC)
        }
        #pragma acc wait
    #endif

    gettimeofday(&end, 0);
    elapsed_update_weights += (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) * 1e-6;
}
