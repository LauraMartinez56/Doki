#include "qstate.h"
#include "platform.h"
#include <mpi.h>        /* AÑADIDO: librería MPI para comunicación entre procesos */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

unsigned char
state_init (struct state_vector *this, unsigned int num_qubits, int init)
{
  size_t i, offset, errored_chunk;
  bool errored;

  if (num_qubits > MAX_NUM_QUBITS)
    {
      return 3;
    }
  this->size = NATURAL_ONE << num_qubits;
  this->fcarg_init = 0;
  this->fcarg = -10.0;
  this->num_qubits = num_qubits;
  this->norm_const = 1;
  this->num_chunks = this->size / COMPLEX_ARRAY_SIZE;
  offset = this->size % COMPLEX_ARRAY_SIZE;
  if (offset > 0)
    {
      this->num_chunks++;
    }
  else
    {
      offset = COMPLEX_ARRAY_SIZE;
    }
  this->vector = MALLOC_TYPE (this->num_chunks, COMPLEX_TYPE *);
  if (this->vector == NULL)
    {
      return 1;
    }
  errored = 0;
  for (i = 0; i < this->num_chunks - 1; i++)
    {
      if (init)
        {
          this->vector[i] = CALLOC_TYPE (COMPLEX_ARRAY_SIZE, COMPLEX_TYPE);
        }
      else
        {
          this->vector[i] = MALLOC_TYPE (COMPLEX_ARRAY_SIZE, COMPLEX_TYPE);
        }
      if (this->vector[i] == NULL)
        {
          errored_chunk = i;
          errored = 1;
          break;
        }
    }
  if (!errored)
    {
      if (init)
        {
          this->vector[this->num_chunks - 1]
              = CALLOC_TYPE (offset, COMPLEX_TYPE);
        }
      else
        {
          this->vector[this->num_chunks - 1]
              = MALLOC_TYPE (offset, COMPLEX_TYPE);
        }
      if (this->vector[this->num_chunks - 1] == NULL)
        {
          errored = 1;
          errored_chunk = this->num_chunks - 1;
        }
    }
  if (errored)
    {
      for (i = 0; i < errored_chunk; i++)
        {
          free (this->vector[i]);
        }
      free (this->vector);
      return 2;
    }
  if (init)
    {
      this->vector[0][0] = COMPLEX_ONE;
    }

  /* AÑADIDO: inicializar la distribución MPI del vector de estado */
  state_init_mpi (this, init);

  return 0;
}

unsigned char
state_clone (struct state_vector *dest, struct state_vector *source)
{
  NATURAL_TYPE i;
  unsigned char exit_code;
  exit_code = state_init (dest, source->num_qubits, 0);
  if (exit_code != 0)
    {
      return exit_code;
    }
  /* MODIFICADO: añadido firstprivate(COMPLEX_ARRAY_SIZE) para evitar
   * error de OpenMP con variables de ámbito no especificado */
#pragma omp parallel for default(none) \
    shared(source, dest, exit_code) \
    firstprivate(COMPLEX_ARRAY_SIZE) \
    private(i)
  for (i = 0; i < source->size; i++)
    {
      dest->vector[i / COMPLEX_ARRAY_SIZE][i % COMPLEX_ARRAY_SIZE]
          = state_get (source, i);
    }
  return 0;
}

void
state_clear (struct state_vector *this)
{
  size_t i;
  if (this->vector != NULL)
    {
      for (i = 0; i < this->num_chunks; i++)
        {
          free (this->vector[i]);
        }
      free (this->vector);
    }
  /* AÑADIDO: liberar el vector local MPI para evitar memory leaks */
  if (this->local_vector != NULL)
    {
      free (this->local_vector);
    }
  this->vector = NULL;
  this->local_vector = NULL;   /* AÑADIDO */
  this->num_chunks = 0;
  this->num_qubits = 0;
  this->size = 0;
  this->global_size = 0;       /* AÑADIDO */
  this->local_size = 0;        /* AÑADIDO */
  this->norm_const = 0.0;
}

void
state_set (struct state_vector *this, NATURAL_TYPE i, COMPLEX_TYPE value)
{
  this->vector[i / COMPLEX_ARRAY_SIZE][i % COMPLEX_ARRAY_SIZE] = value;
}

COMPLEX_TYPE
state_get (struct state_vector *this, NATURAL_TYPE i)
{
  COMPLEX_TYPE val = COMPLEX_DIV_R (
      this->vector[i / COMPLEX_ARRAY_SIZE][i % COMPLEX_ARRAY_SIZE],
      this->norm_const);
  return fix_value (val, -1, -1, 1, 1);
}

size_t
state_mem_size (struct state_vector *this)
{
  size_t state_size;
  if (this == NULL)
    {
      return 0;
    }
  state_size = sizeof (struct state_vector);
  state_size += this->num_chunks * sizeof (COMPLEX_TYPE *);
  state_size
      += (this->num_chunks - 1) * COMPLEX_ARRAY_SIZE * sizeof (COMPLEX_TYPE);
  state_size += (this->size % COMPLEX_ARRAY_SIZE) * sizeof (COMPLEX_TYPE);
  return state_size;
}

/* ── NUEVAS FUNCIONES MPI ── */

/* state_init_mpi: inicializa los campos MPI del struct state_vector.
 * Obtiene el rank y el número de procesos, calcula cuántos elementos
 * corresponden a cada proceso y reserva el vector local. */
void
state_init_mpi (struct state_vector *this, int init)
{
  NATURAL_TYPE rem;
  MPI_Comm_rank (MPI_COMM_WORLD, &this->rank);
  MPI_Comm_size (MPI_COMM_WORLD, &this->nprocs);
  this->global_size = this->size;

  /* Distribución cíclica: local_size = global_size / nprocs
     Los primeros 'rem' procesos reciben un elemento extra */
  rem = this->global_size % this->nprocs;
  if (this->rank < (int)rem)
    this->local_size = this->global_size / this->nprocs + 1;
  else
    this->local_size = this->global_size / this->nprocs;

  /* Reservar memoria para el vector local de este proceso */
  this->local_vector = malloc (this->local_size * sizeof (COMPLEX_TYPE));
  /* Inicializar a cero */
  for (NATURAL_TYPE i = 0; i < this->local_size; i++)
    this->local_vector[i] = COMPLEX_ZERO;

  /* Inicializar todo el vector a cero para todos los procesos */
  for (NATURAL_TYPE j = 0; j < this->size; j++)
    this->vector[j / COMPLEX_ARRAY_SIZE][j % COMPLEX_ARRAY_SIZE] = COMPLEX_ZERO;

  /* Solo el proceso 0 tiene la amplitud 1 en la posicion 0 */
  if (init && this->rank == 0)
    this->vector[0][0] = 1.0 + 0.0 * I;

  printf ("Soy proc %d, tam global %lld, tam local %lld\n",
          this->rank, (long long)this->global_size,
          (long long)this->local_size);
}

/* pdget: obtiene el valor de la amplitud i del vector distribuido.
 * Calcula qué proceso tiene el índice i. Si es local lo devuelve
 * directamente. Si es remoto lo pide con MPI_Send/MPI_Recv. */
COMPLEX_TYPE
pdget (struct state_vector *this, NATURAL_TYPE i)
{
  NATURAL_TYPE local_i;
  int owner;

  /*Distribución cíclica*/

  owner = (int)(i % this->nprocs);
  local_i = i / this->nprocs;
  
  if (owner == this->rank)
    {
      /* Dato local: acceso directo al vector */
      return COMPLEX_DIV_R (
          this->vector[local_i / COMPLEX_ARRAY_SIZE][local_i % COMPLEX_ARRAY_SIZE],
          this->norm_const);
    }
  else
    {
      /* Dato remoto: pedírselo al proceso owner */
      COMPLEX_TYPE val;
      MPI_Send (&i, 1, MPI_INT64_T, owner, 0, MPI_COMM_WORLD);
      MPI_Recv (&val, 1, MPI_DOUBLE_COMPLEX, owner, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      return val;
    }

}

/* pdset: escribe un valor en la posición i del vector distribuido.
 * Solo actúa si el índice pertenece a este proceso. */
void
pdset (struct state_vector *this, NATURAL_TYPE i, COMPLEX_TYPE value)
{
  NATURAL_TYPE local_i;
  int owner;
  
  /* Distribución cíclica */
  owner   = (int)(i % this->nprocs);
  local_i = i / this->nprocs;
  
  if (owner == this->rank)
    {
      /* Dato local: escribir directamente */
      this->vector[local_i / COMPLEX_ARRAY_SIZE][local_i % COMPLEX_ARRAY_SIZE] = value;
    }
  else
    {
      /* Dato remoto: enviarlo al proceso owner */
      MPI_Send (&value, 1, MPI_DOUBLE_COMPLEX, owner, 3, MPI_COMM_WORLD);
    }
}

/* pdgather: recopila el vector completo en todos los procesos.
 * Cada proceso comparte su parte con todos los demás usando MPI_Allgather.
 * Devuelve un array con el vector completo que hay que liberar con free(). */
COMPLEX_TYPE *
pdgather (struct state_vector *this)
{
  COMPLEX_TYPE *full_vector = calloc (this->global_size, sizeof (COMPLEX_TYPE));
  
  if (this->rank == 0)
    {
      /* Proceso 0: colocar sus propios elementos */
      for (NATURAL_TYPE i = 0; i < this->local_size; i++)
        {
          NATURAL_TYPE global_i = i * this->nprocs; /* distribución cíclica */
          full_vector[global_i] = this->vector[i / COMPLEX_ARRAY_SIZE][i % COMPLEX_ARRAY_SIZE];
        }
      /* Recibir elementos del resto de procesos */
      for (int p = 1; p < this->nprocs; p++)
        {
          NATURAL_TYPE remote_size = this->global_size / this->nprocs;
          if (p < (int)(this->global_size % this->nprocs))
            remote_size++;
          COMPLEX_TYPE *buf = malloc (remote_size * sizeof (COMPLEX_TYPE));
          MPI_Recv (buf, (int)remote_size, MPI_DOUBLE_COMPLEX, p, 0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (NATURAL_TYPE i = 0; i < remote_size; i++)
            {
              NATURAL_TYPE global_i = i * this->nprocs + p;
              full_vector[global_i] = buf[i];
            }
          free (buf);
        }
    }
  else
    {
      /* Resto de procesos: enviar sus elementos al proceso 0 */
      COMPLEX_TYPE *send_buf = malloc (this->local_size * sizeof (COMPLEX_TYPE));
      for (NATURAL_TYPE i = 0; i < this->local_size; i++)
        send_buf[i] = this->vector[i / COMPLEX_ARRAY_SIZE][i % COMPLEX_ARRAY_SIZE];
      MPI_Send (send_buf, (int)this->local_size, MPI_DOUBLE_COMPLEX, 0, 0,
                MPI_COMM_WORLD);
      free (send_buf);
    }

  /* Difundir el vector completo a todos los procesos*/
  if (this->rank == 0)
    {
      for (int p = 1; p < this->nprocs; p++)
        MPI_Send (full_vector, (int)this->global_size, MPI_DOUBLE_COMPLEX, p, 1,
                  MPI_COMM_WORLD);
    }
  else
    {
      MPI_Recv (full_vector, (int)this->global_size, MPI_DOUBLE_COMPLEX, 0, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  return full_vector;
}
