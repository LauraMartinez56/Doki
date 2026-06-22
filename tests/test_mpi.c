#include "qstate.h"
#include "qgate.h"
#include "qops.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* Test 1: state_init_mpi */
    printf("=== Test state_init_mpi ===\n");
    struct state_vector sv;
    unsigned char result = state_init(&sv, 3, 1);
    if (result != 0)
        printf("Error en state_init: %d\n", result);

    /* Test 2: pdget */
    printf("=== Test pdget ===\n");
    for (int i = 0; i < 8; i++)
        printf("pdget(%d) = %f + %fi\n", i,
               __real__ pdget(&sv, i),
               __imag__ pdget(&sv, i));

    /* Test 3: pdset */
    printf("=== Test pdset ===\n");
    pdset(&sv, 1, 0.5 + 0.0 * I);
    printf("pdget(1) tras pdset = %f + %fi\n",
           __real__ pdget(&sv, 1),
           __imag__ pdget(&sv, 1));

    /* Restaurar estado inicial */
    pdset(&sv, 1, 0.0 + 0.0 * I);
    pdset(&sv, 0, 1.0 + 0.0 * I);

    /* Test 4: apply_gate con puerta Hadamard */
    printf("=== Test apply_gate (Hadamard) ===\n");

    struct qgate h_gate;
    h_gate.num_qubits = 1;
    h_gate.size = 2;
    h_gate.matrix = malloc(2 * sizeof(COMPLEX_TYPE *));
    h_gate.matrix[0] = malloc(2 * sizeof(COMPLEX_TYPE));
    h_gate.matrix[1] = malloc(2 * sizeof(COMPLEX_TYPE));

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    h_gate.matrix[0][0] = inv_sqrt2 + 0.0 * I;
    h_gate.matrix[0][1] = inv_sqrt2 + 0.0 * I;
    h_gate.matrix[1][0] = inv_sqrt2 + 0.0 * I;
    h_gate.matrix[1][1] = -inv_sqrt2 + 0.0 * I;

    unsigned int targets[1] = {0};
    struct state_vector new_sv;
    result = apply_gate(&sv, &h_gate, targets, 1, NULL, 0, NULL, 0, &new_sv);

    if (result != 0) {
        printf("Error en apply_gate: %d\n", result);
    } else {
        printf("Estado tras Hadamard en qubit 0:\n");
        for (int i = 0; i < 8; i++)
            printf("pdget(%d) = %f + %fi\n", i,
                   __real__ pdget(&new_sv, i),
                   __imag__ pdget(&new_sv, i));
    }

    free(h_gate.matrix[0]);
    free(h_gate.matrix[1]);
    free(h_gate.matrix);
    state_clear(&sv);
    state_clear(&new_sv);

    MPI_Finalize();
    return 0;
}
