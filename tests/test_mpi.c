#include "qstate.h"
#include "qgate.h"
#include "qops.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

/* Construye la puerta X (NOT) */
struct qgate build_X()
{
    struct qgate x;
    x.num_qubits = 1;
    x.size = 2;
    x.matrix = malloc(2 * sizeof(COMPLEX_TYPE *));
    x.matrix[0] = malloc(2 * sizeof(COMPLEX_TYPE));
    x.matrix[1] = malloc(2 * sizeof(COMPLEX_TYPE));
    x.matrix[0][0] = 0.0 + 0.0 * I;
    x.matrix[0][1] = 1.0 + 0.0 * I;
    x.matrix[1][0] = 1.0 + 0.0 * I;
    x.matrix[1][1] = 0.0 + 0.0 * I;
    return x;
}

/* Construye la puerta Z (Phase flip) */
struct qgate build_Z()
{
    struct qgate z;
    z.num_qubits = 1;
    z.size = 2;
    z.matrix = malloc(2 * sizeof(COMPLEX_TYPE *));
    z.matrix[0] = malloc(2 * sizeof(COMPLEX_TYPE));
    z.matrix[1] = malloc(2 * sizeof(COMPLEX_TYPE));
    z.matrix[0][0] = 1.0 + 0.0 * I;
    z.matrix[0][1] = 0.0 + 0.0 * I;
    z.matrix[1][0] = 0.0 + 0.0 * I;
    z.matrix[1][1] = -1.0 + 0.0 * I;
    return z;
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

void test_gate(int num_qubits, unsigned int target_qubit,
               NATURAL_TYPE init_pos, struct qgate *gate,
               const char *gate_name, int rank)
{
    if (rank == 0)
        printf("\n=== %d qubits, estado inicial |%lld>, puerta %s en qubit %d ===\n",
               num_qubits, (long long)init_pos, gate_name, target_qubit);
    struct state_vector sv;
    state_init(&sv, num_qubits, 1);
    state_init_mpi(&sv, 1);
    if (init_pos != 0) {
        pdset(&sv, 0, 0.0 + 0.0 * I);
        pdset(&sv, init_pos, 1.0 + 0.0 * I);
    }
    if (rank == 0) printf("--- Estado inicial ---\n");
    print_state(&sv, rank, "state");
    struct state_vector new_sv;
    apply_gate(&sv, gate, &target_qubit, 1, NULL, 0, NULL, 0, &new_sv);
    if (rank == 0) printf("--- Estado tras %s ---\n", gate_name);
    print_state(&new_sv, rank, "new_state");
    state_clear(&sv);
    state_clear(&new_sv);
}
void test_tiempo(int num_qubits, int rank)
{
    struct state_vector sv;
    state_init(&sv, num_qubits, 1);
    state_init_mpi(&sv, 1);
    struct qgate h = build_hadamard();
    struct state_vector new_sv;
    unsigned int target = 0;
    double t_start = MPI_Wtime();
    apply_gate(&sv, &h, &target, 1, NULL, 0, NULL, 0, &new_sv);
    double t_end = MPI_Wtime();
    if (rank == 0)
        printf("=== %d qubits: tiempo = %f segundos ===\n", num_qubits, t_end - t_start);
    free(h.matrix[0]); free(h.matrix[1]); free(h.matrix);
    state_clear(&sv); state_clear(&new_sv);
    sleep(1);
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

    /* Test superposicion completa */
if (rank == 0) printf("\n=== 3 qubits, superposicion completa (H en todos los qubits) ===\n");
struct state_vector sv3;
state_init(&sv3, 3, 1);
state_init_mpi(&sv3, 1);
struct qgate h2 = build_hadamard();
struct state_vector sv3b, sv3c, sv3d;
apply_gate(&sv3, &h2, (unsigned int[]){0}, 1, NULL, 0, NULL, 0, &sv3b);
apply_gate(&sv3b, &h2, (unsigned int[]){1}, 1, NULL, 0, NULL, 0, &sv3c);
apply_gate(&sv3c, &h2, (unsigned int[]){2}, 1, NULL, 0, NULL, 0, &sv3d);
if (rank == 0) printf("--- Estado tras H en todos los qubits ---\n");
print_state(&sv3d, rank, "state");
free(h2.matrix[0]); free(h2.matrix[1]); free(h2.matrix);
state_clear(&sv3); state_clear(&sv3b); state_clear(&sv3c); state_clear(&sv3d);
sleep(1);

/* Test escalabilidad */
test_hadamard_pos(7, 0, 0, rank);
sleep(1);
/* Tests de tiempo para gráficas */
test_tiempo(3, rank);
test_tiempo(5, rank);
test_tiempo(7, rank);
test_tiempo(10, rank);
test_tiempo(12, rank);
test_tiempo(14, rank);
test_tiempo(16, rank);

    /* Tests puerta X */
    struct qgate x = build_X();
    test_gate(3, 0, 0, &x, "X", rank);
    test_gate(3, 0, 1, &x, "X", rank);
    free(x.matrix[0]); free(x.matrix[1]); free(x.matrix);

    sleep(1);

    /* Superposicion completa + puerta X */
if (rank == 0) printf("\n=== 3 qubits, superposicion completa + puerta X en qubit 0 ===\n");
struct state_vector sv_x, sv_x2;
state_init(&sv_x, 3, 1);
state_init_mpi(&sv_x, 1);
struct qgate hx = build_hadamard();
apply_gate(&sv_x, &hx, (unsigned int[]){0}, 1, NULL, 0, NULL, 0, &sv_x2);
free(hx.matrix[0]); free(hx.matrix[1]); free(hx.matrix);
struct qgate xx = build_X();
struct state_vector sv_x3;
apply_gate(&sv_x2, &xx, (unsigned int[]){0}, 1, NULL, 0, NULL, 0, &sv_x3);
free(xx.matrix[0]); free(xx.matrix[1]); free(xx.matrix);
if (rank == 0) printf("--- Estado tras H + X ---\n");
print_state(&sv_x3, rank, "state");
state_clear(&sv_x); state_clear(&sv_x2); state_clear(&sv_x3);
sleep(1);


/* Superposicion completa + puerta Z */
if (rank == 0) printf("\n=== 3 qubits, superposicion completa + puerta Z en qubit 0 ===\n");
struct state_vector sv_z, sv_z2;
state_init(&sv_z, 3, 1);
state_init_mpi(&sv_z, 1);
struct qgate hz = build_hadamard();
apply_gate(&sv_z, &hz, (unsigned int[]){0}, 1, NULL, 0, NULL, 0, &sv_z2);
free(hz.matrix[0]); free(hz.matrix[1]); free(hz.matrix);
struct qgate zz = build_Z();
struct state_vector sv_z3;
apply_gate(&sv_z2, &zz, (unsigned int[]){0}, 1, NULL, 0, NULL, 0, &sv_z3);
free(zz.matrix[0]); free(zz.matrix[1]); free(zz.matrix);
if (rank == 0) printf("--- Estado tras H + Z ---\n");
print_state(&sv_z3, rank, "state");
state_clear(&sv_z); state_clear(&sv_z2); state_clear(&sv_z3);
sleep(1);

    /* Tests puerta Z */
    struct qgate z = build_Z();
    test_gate(3, 0, 0, &z, "Z", rank);
    test_gate(3, 0, 1, &z, "Z", rank);
    free(z.matrix[0]); free(z.matrix[1]); free(z.matrix);

    MPI_Finalize();
    return 0;
}
