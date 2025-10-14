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

/**
 * @brief Iniciatlitza la capa incial de la xarxa (input layer) amb l'entrada
 * que volem reconeixer.
 *
 * @param i Índex de l'element del conjunt d'entrenament que farem servir.
 **/
void feed_input(int i) {
    // Natan (2025-10-14): removed parallelization (see results.md#feed_input)
    // #pragma omp parallel for  // Natan (2025-10-09): parallelized the copy from input to first layer
    for (int j = 0; j < num_neurons[0]; j++)
        lay[0].actv[j] = input[i][j]; // copy input from pattern i (0s and 1s) to input layer
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
void forward_prop() { // z_j^(i) = b_j^(i) + sum_{k=0}^{n_{i-1}-1} w_{jk}^{(i-1)} * a_k^{(i-1)}
    for (int i = 1; i < num_layers; i++) {
        #pragma omp parallel for  // Natan (2025-10-09): parallelized the computation of each neuron in the layer
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].z[j] = lay[i].bias[j];
            for (int k = 0; k < num_neurons[i - 1]; k++)
                lay[i].z[j] +=
                    ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) * // weight from neuron k in layer i-1 to neuron j in layer i, indexed in a 1D array as(weightn1->n1, weightn2->n1, weightn3->n1, weightn1->n2, weightn2->n2, weightn3->n2,...)
                     (lay[i - 1].actv[k]));

            if (i <
                num_layers - 1)  // Relu Activation Function for Hidden Layers
                lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
            else  // Sigmoid Activation Function for Output Layer
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
   #pragma omp parallel for // Ferran (2025-10-13): parallelized the computation of each neuron in the output layer
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

    /*
    En aquí la qüestió és que només es fa una iteració entre la última i la penúltima layer (ara separat en dw i dactv),
    si se'n fes entre diverses layers, i intervingués un bucle for afegit a dalt de tot tindríem
    que el dactiv sí que s'acumularia (però això ja passa i es tracta en el hidden layers)
    
    Codi inicial:
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
                (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
            lay[num_layers - 2].dactv[k] =
                lay[num_layers - 2]
                    .out_weights[j * num_neurons[num_layers - 2] + k] *
                lay[num_layers - 1].dz[j];
        }
    }
    *
    // Calcular dw
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
                lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k];
        }
    }
    // Calcular dactv
    for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
        double suma = 0.0;
        #pragma omp parallel for reduction(+:suma)
        for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
            suma += lay[num_layers - 2].out_weights[j * num_neurons[num_layers - 2] + k] *
                    lay[num_layers - 1].dz[j];
        }
        lay[num_layers - 2].dactv[k] = suma;
    }
        */
    
    for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
        for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
            lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
                (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
            lay[num_layers - 2].dactv[k] =
                lay[num_layers - 2]
                    .out_weights[j * num_neurons[num_layers - 2] + k] *
                lay[num_layers - 1].dz[j];
        }
    }
    
    // Hidden Layers
    for (int i = num_layers - 2; i > 0; i--) {
        #pragma omp parallel for  // Natan (2025-10-09): parallelized the backpropagation for each neuron in the layer
        for (int j = 0; j < num_neurons[i]; j++) {
            lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0; // dactv ja conté l'error propagat de la capa següent, i aquest es multiplica per la derivada de ReLU amb la fórmula matemàtica δ_j = ∂E/∂z_j =  ∂E/∂a_j * ∂a_j/∂z_j

            for (int k = 0; k < num_neurons[i - 1]; k++) {
                lay[i - 1].dw[j * num_neurons[i - 1] + k] =
                    lay[i].dz[j] * lay[i - 1].actv[k];

                if (i > 1){ // No cal propagar l'error més enllà de la capa oculta més propera a la capa d'entrada, ja que aquesta lay[0] és coneguda (imatge d'entrada)
                     //dactv[k] és compartit entre totes les iteracions de j
                    //#pragma omp atomic // Ferran (2025-10-13): avoid race condition
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
        #pragma omp parallel for  // Natan (2025-10-09): parallelized the update of weights
        for (int j = 0; j < num_neurons[i + 1]; j++)
            for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                lay[i].out_weights[j * num_neurons[i] + k] =
                    (lay[i].out_weights[j * num_neurons[i] + k]) -
                    (alpha * lay[i].dw[j * num_neurons[i] + k]);
        #pragma omp parallel for  // Ferran (2025-10-13): parallelized the update of biases
        for (int j = 0; j < num_neurons[i]; j++)  // Update Bias
            lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);
    }
}
