#pragma once
#ifndef QSTATE_H_
#define QSTATE_H_

#include "platform.h"
#include <mpi.h>
#include <stdbool.h>

struct state_vector
{
  /* total size of the vector */
  NATURAL_TYPE size;
  /* number of chunks */
  size_t num_chunks;
  /* number of qubits in this quantum system */
  unsigned int num_qubits;
  /* partial vector */
  COMPLEX_TYPE **vector;
  /* normalization constant */
  REAL_TYPE norm_const;
  /* fcarg initialized */
  bool fcarg_init;
  /* first complex argument */
  REAL_TYPE fcarg;
  /* MPI: tamaño global del vector */
  NATURAL_TYPE global_size;
  /* MPI: tamaño local de este proceso */
  NATURAL_TYPE local_size;
  /* MPI: vector local plano */
  COMPLEX_TYPE *local_vector;
  /* MPI: rank de este proceso */
  int rank;
  /* MPI: número total de procesos */
  int nprocs;
};

unsigned char state_init (struct state_vector *this, unsigned int num_qubits,
                          int init);
unsigned char state_clone (struct state_vector *dest,
                           struct state_vector *source);
void state_clear (struct state_vector *this);
void state_set (struct state_vector *this, NATURAL_TYPE i, COMPLEX_TYPE value);
COMPLEX_TYPE state_get (struct state_vector *this, NATURAL_TYPE i);
size_t state_mem_size (struct state_vector *this);

/* Nuevas funciones MPI */
void state_init_mpi (struct state_vector *this);
COMPLEX_TYPE pdget (struct state_vector *this, NATURAL_TYPE i);
void pdset (struct state_vector *this, NATURAL_TYPE i, COMPLEX_TYPE value);

#endif /* QSTATE_H_ */
