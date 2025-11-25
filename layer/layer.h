/*
 *  layer.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: fall 21 (curs 21-22)
 *  Last modified: fall 24 (curs 24-25)
 *  Author: Blanca Llauradó,
 *  Modified: Christian Germer
 *
 *  Descripció:
 *  Constants, estructures de dades i capçaleres de les funcions auxiliars per la creació de cada capa (layer) de la xarxa neuronal.
 *
 */

#ifndef LAYER_H
#define LAYER_H

typedef struct layer_t {
    int num_neu;
    float* actv;
    float* bias;
    float* z;
    float* dactv;
    float* dbias;
    float* dz;
    float* out_weights;
    float* dw;
} layer;

extern int debug;

layer create_layer(int num_neurons, int number_of_neurons_next_layer);
void free_layer(layer lay);

#endif

/*
w[j*k]    : w[j*k] - alpha * dw[j*k];
actv[j]   : activació de la neurona j de la capa actual (a_j = f(z_j))
            capa oculta (ReLU): a_j = max(0, z_j)
            capa de sortida (sigmoid): a_j = 1 / (1 + exp(-z_j))
bias[j]   : b_j = b_j - alpha * dbias_j
z[j]      : excitació de la neurona abans de la funció d'activació
            z_j = sum_k w_jk * a_k_prev + b_j

dz[j]     : error local de la neurona j
            capa de sortida (sigmoid + MSE):
                dz_j = (a_j - y_j) * a_j * (1 - a_j)
            capa oculta (ReLU):
                dz_j = dactv_j * f'(z_j)
                f'(z_j) = 1 si z_j >= 0, 0 si z_j < 0

dbias[j]  : gradient del bias
            dbias_j = dz_j

dw[j*k]   : gradient del pes que connecta neurona k de la capa anterior amb neurona j de la capa actual
            dw_jk = dz_j * a_k_prev

out_weights[j*k] : pes que connecta neurona k de la capa anterior amb neurona j de la capa actual

dactv[k]  : error de la neurona k de la capa anterior (propagat cap enrere)
            dactv_k = sum_j w_jk * dz_j

Resum de direcció:
- actv[i] és la sortida de la capa anterior
- dz[i] i dbias[i] pertanyen a la capa actual
- dw[i] i dactv[i] es fan servir per ajustar la capa anterior

*/
