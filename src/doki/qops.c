#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "platform.h"
#include "qgate.h"
#include "qops.h"
#include "qstate.h"

REAL_TYPE get_global_phase(struct state_vector *state)
{
NATURAL_TYPE i;
REAL_TYPE phase;
COMPLEX_TYPE val;

if (state->fcarg_init) {
return state->fcarg;
}

phase = 0.0;
for (i = 0; i < state->size; i++) {
val = pdget(state, i);
if (RE(val) != 0. || IM(val) != 0.) {
if (IM(val) != 0.) {
phase = ARG(val);
}
break;
}
}
state->fcarg = phase;
state->fcarg_init = 1;

return phase;
}

REAL_TYPE probability(struct state_vector *state, unsigned int target_id)
{
NATURAL_TYPE i, index, qty, low, high, target;
REAL_TYPE value;
COMPLEX_TYPE val;

qty = state->size >> 1;
target = NATURAL_ONE << target_id;
low = target - 1;
high = ~low;

value = 0;
#pragma omp parallel for reduction (+:value) \
default(none) \
firstprivate(state, qty, low, high, target, COMPLEX_ARRAY_SIZE) \
private(i, index, val)
for (i = 0; i < qty; i++) {
index = ((i & high) << 1) + target + (i & low);
val = pdget(state, index);
value += RE(val) * RE(val) + IM(val) * IM(val);
}

return value;
}

unsigned char join(struct state_vector *r, struct state_vector *s1,
   struct state_vector *s2)
{
NATURAL_TYPE i, j, new_index;
COMPLEX_TYPE o1, o2;
unsigned char exit_code;

exit_code = state_init(r, s1->num_qubits + s2->num_qubits, false);
if (exit_code != 0) {
return exit_code;
}

#pragma omp parallel for default(none) \
firstprivate(r, s1, s2, exit_code, COMPLEX_ARRAY_SIZE) \
private(i, j, o1, o2, new_index)
for (i = 0; i < s1->size; i++) {
o1 = pdget(s1, i);
for (j = 0; j < s2->size; j++) {
new_index = i * s2->size + j;
o2 = pdget(s2, j);
pdset(r, new_index, COMPLEX_MULT(o1, o2));
}
}

return 0;
}

unsigned char measure(struct state_vector *state, bool *result,
      unsigned int target, struct state_vector *new_state,
      REAL_TYPE roll)
{
REAL_TYPE sum;
unsigned char exit_code;

sum = probability(state, target);
*result = sum > roll;
exit_code = collapse(state, target, *result, sum, new_state);

return exit_code;
}

unsigned char collapse(struct state_vector *state, unsigned int target_id,
       bool value, REAL_TYPE prob_one,
       struct state_vector *new_state)
{
unsigned char exit_code;
NATURAL_TYPE i, j, low, high, val;

if (state->num_qubits == 1) {
new_state->vector = NULL;
new_state->num_qubits = 0;
return 0;
}

exit_code = state_init(new_state, state->num_qubits - 1, false);
if (exit_code != 0) {
free(new_state);
return exit_code;
}
val = NATURAL_ONE << target_id;
low = val - 1;
high = ~low;
if (!value) {
prob_one = 1 - prob_one;
val = 0;
}

#pragma omp parallel for default(none) \
firstprivate(state, new_state, low, high, val, COMPLEX_ARRAY_SIZE) \
private(i, j)
for (j = 0; j < new_state->size; j++) {
i = ((j & high) << 1) + val + (j & low);
pdset(new_state, j, pdget(state, i));
}
new_state->norm_const = sqrt(prob_one);

return 0;
}

unsigned char apply_gate(struct state_vector *state, struct qgate *gate,
 unsigned int *targets, unsigned int num_targets,
 unsigned int *controls, unsigned int num_controls,
 unsigned int *anticontrols,
 unsigned int num_anticontrols,
 struct state_vector *new_state)
{
REAL_TYPE norm_const;
unsigned char exit_code;
NATURAL_TYPE control_mask, anticontrol_mask, i, reg_index;
NATURAL_TYPE base_idx, rem_idx, local_start, local_end;
unsigned int j, k, row;
COMPLEX_TYPE sum;
COMPLEX_TYPE *full_vector;

if (new_state == NULL)
return 10;

exit_code = state_init(new_state, state->num_qubits, false);
if (exit_code != 0) {
free(new_state);
return exit_code;
}

control_mask = NATURAL_ZERO;
for (j = 0; j < num_controls; j++)
control_mask |= NATURAL_ONE << controls[j];
anticontrol_mask = NATURAL_ZERO;
for (j = 0; j < num_anticontrols; j++)
anticontrol_mask |= NATURAL_ONE << anticontrols[j];

/* Recopilar vector completo en todos los procesos */
full_vector = pdgather(state);

/* Calcular rango local de este proceso */
base_idx = state->global_size / state->nprocs;
rem_idx  = state->global_size % state->nprocs;
if (state->rank < (int)rem_idx) {
local_start = state->rank * (base_idx + 1);
local_end   = local_start + base_idx + 1;
} else {
local_start = rem_idx * (base_idx + 1) +
      (state->rank - rem_idx) * base_idx;
local_end   = local_start + base_idx;
}

norm_const = 0;
for (i = local_start; i < local_end; i++) {
if ((i & control_mask) == control_mask &&
    (i & anticontrol_mask) == 0) {
sum = COMPLEX_ZERO;
reg_index = i;
for (j = 0; j < gate->size; j++) {
row = 0;
for (k = 0; k < num_targets; k++) {
row += ((i & (NATURAL_ONE
      << targets[k])) != 0)
       << k;
if ((j & (NATURAL_ONE << k)) != 0)
reg_index |= NATURAL_ONE
     << targets[k];
else
reg_index &= ~(NATURAL_ONE
       << targets[k]);
}
sum = COMPLEX_ADD(sum,
COMPLEX_MULT(
COMPLEX_DIV_R(full_vector[reg_index],
      state->norm_const),
gate->matrix[row][j]));
}
} else {
sum = COMPLEX_DIV_R(full_vector[i], state->norm_const);
}
pdset(new_state, i, sum);
norm_const += pow(RE(sum), 2) + pow(IM(sum), 2);
}
free(full_vector);
/* Reducir norm_const entre todos los procesos */
REAL_TYPE global_norm;
MPI_Allreduce(&norm_const, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
new_state->norm_const = sqrt(global_norm);

return 0;
}

/*
 * MODIFICACIONES MPI:
 *
 * get_global_phase, probability, join, collapse:
 *   Sustituidos state_get por pdget y state_set por pdset.
 *
 * apply_gate:
 *   Usa pdgather para recopilar el vector completo en todos los procesos.
 *   Cada proceso calcula solo su rango local de indices.
 *   Usa pdset para escribir los resultados en el nuevo estado distribuido.
 */
