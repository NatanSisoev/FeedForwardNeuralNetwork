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

#if defined(TRAIN) || defined(TEST) || defined(ALL)
#include <mpi.h>
#endif

double elapsed_train = 0.0;
double elapsed_test = 0.0;


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
    struct timeval train_begin, train_end;
    gettimeofday(&train_begin, 0);


    // printf("\nTraining...\n");
    
    #if defined(TRAIN) || defined(ALL)
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int extra = num_training_patterns % num_procs;
        int from  = rank * (num_training_patterns / num_procs) + ((rank < extra) ? rank : extra);
        int to    = from + (num_training_patterns / num_procs) + ((rank < extra) ? 1 : 0);

        if (debug)
            printf("TRAIN Rank %d -> from %d to %d\n", rank, from, to);
    #endif

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path,
                                1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    int ranpat[num_training_patterns];

    // Data copy
    #if defined(OPENACC)
    #pragma acc enter data copyin(alpha)
    #pragma acc enter data copyin(num_out_layer)
    #pragma acc enter data copyin(num_layers)
    #pragma acc enter data copyin(num_neurons[0 : num_layers])

    #pragma acc enter data copyin(desired_outputs[0 : num_training_patterns])
    #pragma acc enter data copyin(input[0 : num_training_patterns])
    
    for (int i = 0; i < num_training_patterns; i++) {
       #pragma acc enter data copyin(input[i][0 : num_neurons[0]],   \
                              desired_outputs[i][0 : num_out_layer])
    }

    #pragma acc enter data copyin(lay[0 : num_layers])
    for (int i = 0; i < num_layers; i++) {
    #pragma acc enter data copyin(lay[i].actv[0 : num_neurons[i]], \
                    lay[i].bias[0 : num_neurons[i]], \
                    lay[i].z[0 : num_neurons[i]], \
                    lay[i].dactv[0 : num_neurons[i]], \
                    lay[i].dbias[0 : num_neurons[i]], \
                    lay[i].dz[0 : num_neurons[i]])
    }

    for (int i = 0; i < num_layers - 1; i++) {
    #pragma acc enter data copyin(lay[i].out_weights[0 : num_neurons[i] * num_neurons[i + 1]], \
        lay[i].dw[0 : num_neurons[i] * num_neurons[i + 1]])
    }
    // End data copy
    #endif
    
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
        #if defined(TRAIN) || defined(ALL)
            for (int i = from; i < to; i++) {
                int p = ranpat[i];
                feed_input(p);
                forward_prop();
                back_prop(p);
                update_weights();
            }
            if (debug == 1 && rank == 0)
                printf("Epoch %d finished\n", it);
            
        #else
            for (int i = 0; i < num_training_patterns; i++) {
                int p = ranpat[i];
                feed_input(p);
                forward_prop();
                back_prop(p);
                update_weights();
            }
            if (debug == 1)
                printf("Epoch %d finished\n", it);
        #endif
    }


    #if defined(TRAIN) || defined(ALL)
        for (int l = 0; l < num_layers - 1; l++) {
            long size = num_neurons[l] * num_neurons[l+1];
            MPI_Allreduce(MPI_IN_PLACE, lay[l].out_weights, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            // Mitjana
            for (long i = 0; i < size; i++)
                lay[l].out_weights[i] /= num_procs;

            MPI_Allreduce(MPI_IN_PLACE, lay[l].bias, num_neurons[l+1], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < num_neurons[l+1]; i++)
                lay[l].bias[i] /= num_procs;
        }
    #endif

    #if defined(TRAIN) || defined(ALL)
        double global_feed, global_forward, global_back, global_update;

        MPI_Reduce(&elapsed_feed_input,     &global_feed,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&elapsed_forward_prop,   &global_forward, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&elapsed_back_prop,      &global_back,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&elapsed_update_weights, &global_update,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("TIMES (MPI - wall time):\n");
            printf("FEED_INPUT      %f\n", global_feed);
            printf("FORWARD_PROP    %f\n", global_forward);
            printf("BACK_PROP       %f\n", global_back);
            printf("UPDATE_WEIGHTS  %f\n", global_update);
        }
    #else
        printf("TIMES:\n");
        printf("FEED_INPUT\t%f\n", elapsed_feed_input);
        printf("FORWARD_PROP\t%f\n", elapsed_forward_prop);
        printf("BACK_PROP\t%f\n", elapsed_back_prop);
        printf("UPDATE_WEIGHTS\t%f\n", elapsed_update_weights);
    #endif

    gettimeofday(&train_end, 0);
    long train_seconds = train_end.tv_sec - train_begin.tv_sec;
    long train_microseconds = train_end.tv_usec - train_begin.tv_usec;
    elapsed_train = train_seconds + train_microseconds * 1e-6;

    freeInput(num_training_patterns, input);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    struct timeval test_begin, test_end;
    gettimeofday(&test_begin, 0);

    char** rSet;

    // printf("\nTesting...\n");

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        exit(-1);
    }
    #if defined(TEST) || defined(ALL)
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int extra = num_test_patterns % num_procs;
        int from  = rank * (num_test_patterns / num_procs) + ((rank < extra) ? rank : extra);
        int to    = from + (num_test_patterns / num_procs) + ((rank < extra) ? 1 : 0);

        if (debug)
            printf("TEST Rank %d -> from %d to %d\n", rank, from, to);
    #endif


    #if defined(TEST) || defined(ALL)
        for (int i = from; i < to; i++) {
    #else
        for (int i = 0; i < num_test_patterns; i++) {
    #endif
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];
        
        #if defined(OPENACC)
        #pragma acc update device(lay[0].actv[0:num_neurons[0]])
        #endif
        forward_prop();
        #if defined(OPENACC)
        #pragma acc update host(lay[num_layers - 1].actv[0:num_neurons[num_layers - 1]])
        #endif
        printRecognized(i, lay[num_layers - 1]);
    }

    #if defined(TEST) || defined(ALL)
        int global_total = 0;
        MPI_Allreduce(&total, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        total = global_total;
        if (rank == 0)
            printf("%d\t", total);
    #else
        printf("%d\t", total);
    #endif
    
    gettimeofday(&test_end, 0);
    long test_seconds = test_end.tv_sec - test_begin.tv_sec;
    long test_microseconds = test_end.tv_usec - test_begin.tv_usec;
    elapsed_test = test_seconds + test_microseconds * 1e-6;

    freeInput(num_test_patterns, rSet);
}

//-----------MAIN-----------//
int main(int argc, char** argv) {
    #if defined(TRAIN) || defined(TEST) || defined(ALL)
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);        
    #endif

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

    

    #if defined(OPENACC)
    for (int i = 0; i < num_layers; i++) {
        #pragma acc exit data delete(lay[i].actv[0:num_neurons[i]])
        #pragma acc exit data delete(lay[i].bias[0:num_neurons[i]])
        #pragma acc exit data delete(lay[i].z[0:num_neurons[i]])
        #pragma acc exit data delete(lay[i].dactv[0:num_neurons[i]])
        #pragma acc exit data delete(lay[i].dbias[0:num_neurons[i]])
        #pragma acc exit data delete(lay[i].dz[0:num_neurons[i]])

        if (i < num_layers - 1) {
            long ow = (long)num_neurons[i+1] * (long)num_neurons[i];
            #pragma acc exit data delete(lay[i].out_weights[0:ow])
            #pragma acc exit data delete(lay[i].dw[0:ow])
        }
    }
    #pragma acc exit data delete(lay[0:num_layers])
    for (int i = 0; i < num_out_layer; i++) {
        #pragma acc exit data delete(desired_outputs[i][0:num_out_layer])
    }
    #pragma acc exit data delete(desired_outputs[0:num_out_layer])
    #pragma acc exit data delete(num_neurons[0:num_layers])
    #pragma acc exit data delete(alpha, num_out_layer, num_layers)
    #endif

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    free(cost);



    #if defined(TRAIN) || defined(TEST)  || defined(ALL)
        if(rank == 0) {
            printf("TRAIN_TIME: %f sec\n", elapsed_train);
            printf("TEST_TIME: %f sec\n", elapsed_test);
            printf("TOTAL_TIME: %f sec\n", elapsed);
        }

        MPI_Finalize();
    #else
        printf("TRAIN_TIME: %f sec\n", elapsed_train);
        printf("TEST_TIME: %f sec\n", elapsed_test);
        printf("TOTAL_TIME: %f sec\n", elapsed);
    #endif

    return 0;
}
