#include "qstate.h"
#include "qgate.h"
#include "qops.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Construye la puerta Hadamard */
struct qgate build_hadamard()
{
    struct qgate h;
    h.num_qubits = 1;
    h.size = 2;
    h.matrix = malloc(2 * sizeof(COMPLEX_TYPE *));
    h.matrix[0] = malloc(2 * sizeof(COMPLEX_TYPE));
    h.matrix[1] = malloc(2 * sizeof(COMPLEX_TYPE));
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    h.matrix[0][0] = inv_sqrt2 + 0.0 * I;
    h.matrix[0][1] = inv_sqrt2 + 0.0 * I;
    h.matrix[1][0] = inv_sqrt2 + 0.0 * I;
    h.matrix[1][1] = -inv_sqrt2 + 0.0 * I;
    return h;
}

/* Imprime el estado local de cada proceso */
void print_state(struct state_vector *sv, int rank, const char *label)
{
    for (NATURAL_TYPE local_i = 0; local_i < sv->local_size; local_i++)
    {
        NATURAL_TYPE i = local_i * sv->nprocs + rank; /* distribución cíclica */
        COMPLEX_TYPE val = sv->vector[local_i / COMPLEX_ARRAY_SIZE][local_i % COMPLEX_ARRAY_SIZE];
        printf("[proc %d] %s[%lld] = %f + %fi\n", rank, label,
               (long long)i,
               __real__ val,
               __imag__ val);
    }
}

/* Test: aplica Hadamard a un estado inicial con amplitud 1 en la posicion init_pos */
void test_hadamard_pos(int num_qubits, unsigned int target_qubit,
                       NATURAL_TYPE init_pos, int rank)
{
    if (rank == 0)
        printf("\n=== %d qubits, estado inicial |%lld>, Hadamard en qubit %d ===\n",
               num_qubits, (long long)init_pos, target_qubit);

    /* Inicializar estado a cero */
    struct state_vector sv;
    unsigned char result = state_init(&sv, num_qubits, 1);
    state_init_mpi(&sv, 1);
    if (result != 0) {
        printf("[proc %d] Error en state_init: %d\n", rank, result);
        return;
    }

    /* Poner amplitud 1 en la posicion deseada */
    if (init_pos != 0) {
        pdset(&sv, 0, 0.0 + 0.0 * I);   /* quitar el 1 de la posicion 0 */
        pdset(&sv, init_pos, 1.0 + 0.0 * I);  /* poner el 1 en init_pos */
    }

    if (rank == 0) printf("--- Estado inicial ---\n");
    print_state(&sv, rank, "state");

    struct qgate h = build_hadamard();
    struct state_vector new_sv;
    result = apply_gate(&sv, &h, &target_qubit, 1, NULL, 0, NULL, 0, &new_sv);

    if (result != 0) {
        printf("[proc %d] Error en apply_gate: %d\n", rank, result);
    } else {
        if (rank == 0) printf("--- Estado tras Hadamard ---\n");
        print_state(&new_sv, rank, "new_state");
    }

    free(h.matrix[0]);
    free(h.matrix[1]);
    free(h.matrix);
    state_clear(&sv);
    state_clear(&new_sv);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0)
        printf("Ejecutando con %d proceso(s)\n", nprocs);

    /* Test 1: estado |000> (posicion 0), Hadamard en qubit 0 */
    test_hadamard_pos(3, 0, 0, rank);

    /* Test 2: estado |001> (posicion 1), Hadamard en qubit 0 */
    test_hadamard_pos(3, 0, 1, rank);

    /* Test 3: estado |010> (posicion 2), Hadamard en qubit 1 */
    test_hadamard_pos(3, 1, 2, rank);

    /* Test 4: estado |100> (posicion 4), Hadamard en qubit 2 */
    test_hadamard_pos(3, 2, 4, rank);

    /* Test 5: 5 qubits, estado |00001> (posicion 1), Hadamard en qubit 0 */
    test_hadamard_pos(5, 0, 1, rank);

    MPI_Finalize();
    return 0;
}
