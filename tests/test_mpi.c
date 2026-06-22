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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Test 1: state_init_mpi */
    if (rank == 0) printf("=== Test state_init_mpi ===\n");
    struct state_vector sv;
    unsigned char result = state_init(&sv, 3, 1);
    if (result != 0)
        printf("[proc %d] Error en state_init: %d\n", rank, result);

    /* Test 2: pdget - cada proceso solo lee sus posiciones locales */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("=== Test pdget ===\n");
    MPI_Barrier(MPI_COMM_WORLD);

    NATURAL_TYPE base = sv.global_size / nprocs;
    NATURAL_TYPE rem  = sv.global_size % nprocs;
    NATURAL_TYPE local_start = (rank < (int)rem) ?
        rank * (base + 1) :
        rem * (base + 1) + (rank - rem) * base;
    NATURAL_TYPE local_end = local_start + sv.local_size;

    for (NATURAL_TYPE i = local_start; i < local_end; i++)
        printf("[proc %d] pdget(%lld) = %f + %fi\n", rank, (long long)i,
               __real__ pdget(&sv, i),
               __imag__ pdget(&sv, i));

    
    /* Test 3: apply_gate con puerta Hadamard */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("=== Test apply_gate (Hadamard) ===\n");
    MPI_Barrier(MPI_COMM_WORLD);

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

    MPI_Barrier(MPI_COMM_WORLD);
    if (result != 0) {
        printf("[proc %d] Error en apply_gate: %d\n", rank, result);
    } else {
        NATURAL_TYPE new_local_end = local_start + new_sv.local_size;
        for (NATURAL_TYPE i = local_start; i < new_local_end; i++)
            printf("[proc %d] new_state[%lld] = %f + %fi\n", rank, (long long)i,
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
