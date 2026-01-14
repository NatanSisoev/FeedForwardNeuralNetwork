#include "main.h"
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

void freeInput(int np, char** input) {
    for (int i = 0; i < np; i++)
        free(input[i]);
    free(input);
}

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

void train_neural_net() {
    double t0 = MPI_Wtime();

    int extra = num_training_patterns % num_procs;
    int from  = rank * (num_training_patterns / num_procs) + ((rank < extra) ? rank : extra);
    int to    = from + (num_training_patterns / num_procs) + ((rank < extra) ? 1 : 0);

    if (debug)
        printf("TRAIN Rank %d -> from %d to %d\n", rank, from, to);

    if ((input = loadPatternSet(num_training_patterns, dataset_training_path, 1)) == NULL) {
        printf("Loading Patterns: Error!!\n");
        exit(-1);
    }

    int ranpat[num_training_patterns];
    
    for (int it = 0; it < num_epochs; it++) {
        if (rank == 0) {
            for (int p = 0; p < num_training_patterns; p++)
                ranpat[p] = p;

            for (int p = 0; p < num_training_patterns; p++) {
                int x = rando();
                int np = (x * x) % num_training_patterns;
                int tmp = ranpat[p];
                ranpat[p] = ranpat[np];
                ranpat[np] = tmp;
            }
        }

        MPI_Bcast(ranpat, num_training_patterns, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = from; i < to; i++) {
            int p = ranpat[i];
            feed_input(p);
            forward_prop();
            back_prop(p);
            update_weights();
        }

        for (int l = 0; l < num_layers - 1; l++) {
            long size = num_neurons[l] * num_neurons[l+1];

            MPI_Allreduce(MPI_IN_PLACE, lay[l].out_weights, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (long i = 0; i < size; i++)
                lay[l].out_weights[i] /= num_procs;

            MPI_Allreduce(MPI_IN_PLACE, lay[l].bias, num_neurons[l], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int i = 0; i < num_neurons[l]; i++)
                lay[l].bias[i] /= num_procs;
        }
    }

    freeInput(num_training_patterns, input);

    double local = MPI_Wtime() - t0;
    MPI_Reduce(&local, &train_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

//-----------TEST THE TRAINED NETWORK------------
void test_nn() {
    double t0 = MPI_Wtime();

    char** rSet;

    if ((rSet = loadPatternSet(num_test_patterns, dataset_test_path, 0)) ==
        NULL) {
        printf("Error!!\n");
        exit(-1);
    }

    int extra = num_test_patterns % num_procs;
    int from  = rank * (num_test_patterns / num_procs) + ((rank < extra) ? rank : extra);
    int to    = from + (num_test_patterns / num_procs) + ((rank < extra) ? 1 : 0);

    for (int i = from; i < to; i++) {
        for (int j = 0; j < num_neurons[0]; j++)
            lay[0].actv[j] = rSet[i][j];
        
        forward_prop();
        printRecognized(i, lay[num_layers - 1]);
    }

    int global_total = 0;
    MPI_Allreduce(&total, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total = global_total;
    
    double local = MPI_Wtime() - t0;
    MPI_Reduce(&local, &test_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    freeInput(num_test_patterns, rSet);
}

//-----------MAIN-----------//
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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

    train_neural_net();
    test_nn();

    if (rank == 0) {
        printf("RIGHT %d\n", total);
        printf("ACCUR %6.4f\n", (float) total / num_test_patterns);
        printf("TRAIN %6.4f\n", train_t);
        printf("TEST  %6.4f\n", test_t);
        printf("TOTAL %6.4f\n", train_t + test_t);
    }

    if (dinit() != SUCCESS_DINIT)
        printf("Error in Dinitialization...\n");

    free(cost);

    MPI_Finalize();

    return 0;
}
