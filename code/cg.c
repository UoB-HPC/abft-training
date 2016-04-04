//
// Simple conjugate gradient solver
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void spmv(double *matrix, double *vector, double *output, unsigned N)
{
  for (unsigned y = 0; y < N; y++)
  {
    double tmp = 0.0;
    for (unsigned x = 0; x < N; x++)
    {
      tmp += matrix[x + y*N] * vector[x];
    }
    output[y] = tmp;
  }
}

int main(int argc, char *argv[])
{
  // TODO: Accept command-line arguments for these parameters
  unsigned N = 256;
  unsigned max_itrs = 5000;
  double conv_threshold = 0.000001;

  printf("\n");
  printf("matrix size           = %u x %u\n", N, N);
  printf("maximum iterations    = %u\n", max_itrs);
  printf("convergence threshold = %g\n", conv_threshold);
  printf("\n");

  double *A = malloc(N*N*sizeof(double));
  double *b = malloc(N*sizeof(double));
  double *x = malloc(N*sizeof(double));
  double *r = malloc(N*sizeof(double));
  double *p = malloc(N*sizeof(double));
  double *w = malloc(N*sizeof(double));

  // Initialize symmetric matrix A and vectors b and x
  // TODO: Load values from file
  for (unsigned y = 0; y < N; y++)
  {
    for (unsigned x = y; x < N; x++)
    {
      double value = rand() / (double)RAND_MAX;
      A[x + y*N] = value;
      A[y + x*N] = value;
    }

    b[y] = rand() / (double)RAND_MAX;
    x[y] = 0.0;
  }

  // r = b - Ax;
  // p = r
  spmv(A, x, r, N);
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
    spmv(A, p, w, N);

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

    if (itr % 50 == 0)
      printf("iteration %5u :  rr = %12.4lf\n", itr, rr);
  }

  printf("\n");
  printf("ran for %u iterations\n", itr);

  // Compute Ax
  double *Ax = malloc(N*sizeof(double));
  spmv(A, x, Ax, N);

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
