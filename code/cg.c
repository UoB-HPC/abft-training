//
// Simple conjugate gradient solver
//

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
  double value;
  uint32_t row;
  uint32_t col;
} matrix_entry;

void spmv(matrix_entry *matrix, double *vector, double *output,
          unsigned N, unsigned nnz)
{
  for (unsigned i = 0; i < N; i++)
  {
    output[i] = 0.0;
  }

  for (unsigned i = 0; i < nnz; i++)
  {
    matrix_entry element = matrix[i];

    // TODO: check ECC

    output[element.col] += element.value * vector[element.row];
  }
}

int main(int argc, char *argv[])
{
  // TODO: Accept command-line arguments for these parameters
  unsigned N = 1024;
  unsigned max_itrs = 5000;
  double percentage_nonzero = 0.01;
  double conv_threshold = 0.00001;

  matrix_entry *A = NULL;

  double *b = malloc(N*sizeof(double));
  double *x = malloc(N*sizeof(double));
  double *r = malloc(N*sizeof(double));
  double *p = malloc(N*sizeof(double));
  double *w = malloc(N*sizeof(double));

  // Initialize symmetric sparse matrix A and vectors b and x
  unsigned nnz = 0;
  unsigned allocated = 0;
  for (unsigned y = 0; y < N; y++)
  {
    for (unsigned x = y; x < N; x++)
    {
      // Decide if element should be non-zero
      // Always true for the diagonal
      double p = rand() / (double)RAND_MAX;
      if (p >= (percentage_nonzero - 1.0/N) && x != y)
        continue;

      double value = rand() / (double)RAND_MAX;

      if (nnz+2 > allocated)
      {
        allocated += 512;
        A = realloc(A, allocated*sizeof(matrix_entry));
      }

      matrix_entry element;
      element.value = value;
      element.row   = y;
      element.col   = x;

      // TODO: generate ECC

      A[nnz] = element;
      nnz++;

      if (x == y)
        continue;

      element.value = value;
      element.row   = x;
      element.col   = y;

      // TODO: generate ECC

      A[nnz] = element;
      nnz++;
    }

    b[y] = rand() / (double)RAND_MAX;
    x[y] = 0.0;
  }

  printf("\n");
  printf("matrix size           = %u x %u\n", N, N);
  printf("number of non-zeros   = %u (%.2lf%%)\n", nnz, nnz/(double)(N*N)*100);
  printf("maximum iterations    = %u\n", max_itrs);
  printf("convergence threshold = %g\n", conv_threshold);
  printf("\n");

  // r = b - Ax;
  // p = r
  spmv(A, x, r, N, nnz);
  for (unsigned i = 0; i < N; i++)
  {
    p[i] = r[i] = b[i] - r[i];
  }

  // rr = rT * r
  double rr = 0.0;
  for (unsigned i = 0; i < N; i++)
  {
    rr += r[i] * r[i];
  }

  unsigned itr = 0;
  for (; itr < max_itrs && rr > conv_threshold; itr++)
  {
    // w = A*p
    spmv(A, p, w, N, nnz);

    // pw = pT * A*p
    double pw = 0.0;
    for (unsigned i = 0; i < N; i++)
    {
      pw += p[i] * w[i];
    }

    double alpha = rr / pw;

    // x = x + alpha * p
    // r = r - alpha * A*p
    // rr_new = rT * r
    double rr_new = 0.0;
    for (unsigned i = 0; i < N; i++)
    {
      x[i] += alpha * p[i];
      r[i] -= alpha * w[i];

      rr_new += r[i] * r[i];
    }

    double beta = rr_new / rr;

    // p = r + beta * p
    for (unsigned  i = 0; i < N; i++)
    {
      p[i] = r[i] + beta*p[i];
    }

    rr = rr_new;

    if (itr % 100 == 0)
      printf("iteration %5u :  rr = %12.4lf\n", itr, rr);
  }

  printf("\n");
  printf("ran for %u iterations\n", itr);

  // Compute Ax
  double *Ax = malloc(N*sizeof(double));
  spmv(A, x, Ax, N, nnz);

  // Compare Ax to b
  double err_sq = 0.0;
  double max_err = 0.0;
  for (unsigned i = 0; i < N; i++)
  {
    double err = fabs(b[i] - Ax[i]);
    err_sq += err*err;
    max_err = err > max_err ? err : max_err;
  }
  printf("total error = %lf\n", sqrt(err_sq));
  printf("max error   = %lf\n", max_err);
  printf("\n");

  free(A);
  free(b);
  free(x);
  free(r);
  free(p);
  free(w);
  free(Ax);

  return 0;
}
