#pragma once
#ifndef QOPS_H_
#define QOPS_H_

#include "qgate.h"
#include "qstate.h"
#include <stdbool.h>

unsigned char join(struct state_vector *r, struct state_vector *s1,
   struct state_vector *s2);

unsigned char measure(struct state_vector *state, bool *result,
      unsigned int target, struct state_vector *new_state,
      REAL_TYPE roll);

REAL_TYPE probability(struct state_vector *state, unsigned int target_id);

REAL_TYPE get_global_phase(struct state_vector *state);

unsigned char collapse(struct state_vector *state, unsigned int target_id,
       bool value, REAL_TYPE prob_one,
       struct state_vector *new_state);

unsigned char apply_gate(struct state_vector *state, struct qgate *gate,
 unsigned int *targets, unsigned int num_targets,
 unsigned int *controls, unsigned int num_controls,
 unsigned int *anticontrols,
 unsigned int num_anticontrols,
 struct state_vector *new_state);

#endif /* QOPS_H_ */
