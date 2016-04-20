//
// Simple conjugate gradient solver
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "mmio.h"

#include "common.h"

struct
{
  int    n;              // order of matrix
  int    max_itrs;       // max iterations to run
  double percent_nzero;  // percent of matrix to be non-zero
  double conv_threshold; // convergence threshold to stop CG

  int    inject_bitflip; // flip a random bit in the matrix
} params;

sparse_matrix generate_sparse_matrix(unsigned N, double percent_nonzero);
double        get_timestamp();
void          parse_arguments(int argc, char *argv[]);

sparse_matrix load_sparse_matrix(int *N);

int main(int argc, char *argv[])
{
  parse_arguments(argc, argv);

  //sparse_matrix A = generate_sparse_matrix(params.n, params.percent_nzero);
  sparse_matrix A = load_sparse_matrix(&params.n);

  init_matrix_ecc(A);

  double *b = malloc(params.n*sizeof(double));
  double *x = malloc(params.n*sizeof(double));
  double *r = malloc(params.n*sizeof(double));
  double *p = malloc(params.n*sizeof(double));
  double *w = malloc(params.n*sizeof(double));

  // Initialize vectors b and x
  for (unsigned y = 0; y < params.n; y++)
  {
    b[y] = rand() / (double)RAND_MAX;
    x[y] = 0.0;
  }

  printf("\n");
  printf("matrix size           = %u x %u\n", params.n, params.n);
  printf("number of non-zeros   = %u (%.4f%%)\n",
         A.nnz, A.nnz/((double)params.n*(double)params.n)*100);
  printf("maximum iterations    = %u\n", params.max_itrs);
  printf("convergence threshold = %g\n", params.conv_threshold);
  printf("\n");

  if (params.inject_bitflip)
  {
    // Flip a random bit in a random matrix element
    srand(time(NULL));
    int index = rand() % A.nnz;
    int col   = A.elements[index].col & 0x00FFFFFF;
    int row   = A.elements[index].row & 0x00FFFFFF;
    int bit   = rand() % 128;
    int word  = bit / 32;
    ((uint32_t*)(A.elements+index))[word] ^= 1<<(bit%32);
    printf("*** flipping bit %d of element (%d,%d) ***\n", bit, col, row);

    if (params.inject_bitflip > 1)
    {
      // Flip a second bit immediately before or after the first bit
      bit   = bit < 127 ? bit+1 : bit-1;
      word  = bit / 32;
      ((uint32_t*)(A.elements+index))[word] ^= 1<<(bit%32);
      printf("*** flipping bit %d of (%d,%d) ***\n", bit, col, row);
    }
  }

  double start = get_timestamp();

  // r = b - Ax;
  // p = r
  spmv(A, x, r, params.n);
  for (unsigned i = 0; i < params.n; i++)
  {
    p[i] = r[i] = b[i] - r[i];
  }

  // rr = rT * r
  double rr = 0.0;
  for (unsigned i = 0; i < params.n; i++)
  {
    rr += r[i] * r[i];
  }

  unsigned itr = 0;
  for (; itr < params.max_itrs && rr > params.conv_threshold; itr++)
  {
    // w = A*p
    spmv(A, p, w, params.n);

    // pw = pT * A*p
    double pw = 0.0;
    for (unsigned i = 0; i < params.n; i++)
    {
      pw += p[i] * w[i];
    }

    double alpha = rr / pw;

    // x = x + alpha * p
    // r = r - alpha * A*p
    // rr_new = rT * r
    double rr_new = 0.0;
    for (unsigned i = 0; i < params.n; i++)
    {
      x[i] += alpha * p[i];
      r[i] -= alpha * w[i];

      rr_new += r[i] * r[i];
    }

    double beta = rr_new / rr;

    // p = r + beta * p
    for (unsigned  i = 0; i < params.n; i++)
    {
      p[i] = r[i] + beta*p[i];
    }

    rr = rr_new;

    if (itr % 1 == 0)
      printf("iteration %5u :  rr = %12.4lf\n", itr, rr);
  }

  double end = get_timestamp();

  printf("\n");
  printf("ran for %u iterations\n", itr);

  printf("\ntime taken = %7.2lf ms\n\n", (end-start)*1e-3);

  // Compute Ax
  double *Ax = malloc(params.n*sizeof(double));
  spmv(A, x, Ax, params.n);

  // Compare Ax to b
  double err_sq = 0.0;
  double max_err = 0.0;
  for (unsigned i = 0; i < params.n; i++)
  {
    double err = fabs(b[i] - Ax[i]);
    err_sq += err*err;
    max_err = err > max_err ? err : max_err;
  }
  printf("total error = %lf\n", sqrt(err_sq));
  printf("max error   = %lf\n", max_err);
  printf("\n");

  free(A.elements);
  free(b);
  free(x);
  free(r);
  free(p);
  free(w);
  free(Ax);

  return 0;
}

double get_timestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}

double parse_double(const char *str)
{
  char *next;
  double value = strtod(str, &next);
  return strlen(next) ? -1 : value;
}

int parse_int(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void parse_arguments(int argc, char *argv[])
{
  // Set defaults
  params.n              = 1e6;
  params.max_itrs       = 1000;
  params.percent_nzero  = 0.001;
  params.conv_threshold = 0.001;
  params.inject_bitflip = 0;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--convergence") || !strcmp(argv[i], "-c"))
    {
      if (++i >= argc || (params.conv_threshold = parse_double(argv[i])) < 0)
      {
        printf("Invalid convergence threshold\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (params.max_itrs = parse_int(argv[i])) < 0)
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--norder") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (params.n = parse_int(argv[i])) < 1)
      {
        printf("Invalid matrix order\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--percent-nonzero") || !strcmp(argv[i], "-p"))
    {
      if (++i >= argc || (params.percent_nzero = parse_double(argv[i])) < 0)
      {
        printf("Invalid number of parents\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--inject-bitflip") || !strcmp(argv[i], "-x"))
    {
      params.inject_bitflip = 1;
    }
    else if (!strcmp(argv[i], "--inject-bitflip2") || !strcmp(argv[i], "-xx"))
    {
      params.inject_bitflip = 2;
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./cg [OPTIONS]\n\n");
      printf("Options:\n");
      printf(
        "  -h  --help                 Print this message\n"
        "  -c  --convergence     C    Convergence threshold\n"
        "  -i  --iterations      I    Maximum number of iterations\n"
        "  -n  --norder          N    Order of matrix A\n"
        "  -p  --percent-nzero   P    Percentage of A to be non-zero (approx)\n"
        "  -x  --inject-bitflip       Inject a random bit-flip into A\n"
        "  -xx --inject-bitflip2      Inject a random double bit-flip into A\n"
      );
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }
}

int compare_matrix_elements(const void *a, const void *b)
{
  matrix_entry _a = *(matrix_entry*)a;
  matrix_entry _b = *(matrix_entry*)b;

  if (_a.row < _b.row)
  {
    return -1;
  }
  else if (_a.row > _b.row)
  {
    return 1;
  }
  else
  {
    return _a.col - _b.col;
  }
}

// Load a sparse matrix from a matrix-market format file
sparse_matrix load_sparse_matrix(int *N)
{
  sparse_matrix M = {0, NULL};

  FILE *file = fopen("matrices/shallow_water1/shallow_water1.mtx", "r");
  if (file == NULL)
  {
    printf("Failed to open matrix file\n");
    exit(1);
  }

  int width, height, nnz;
  mm_read_mtx_crd_size(file, &width, &height, &nnz);
  if (width != height)
  {
    printf("Matrix is not square\n");
    exit(1);
  }

  printf("block size = %d x %d with %d nnz\n", width, height, nnz);

  int scale = 100;

  M.nnz = 0;
  M.elements = malloc(scale*2*nnz*sizeof(matrix_entry));
  for (int i = 0; i < nnz; i++)
  {
    matrix_entry element;

    int col, row;

    fscanf(file, "%d %d %lg\n", &col, &row, &element.value);
    col--; /* adjust from 1-based to 0-based */
    row--;

    element.col = col;
    element.row = row;
    M.elements[M.nnz] = element;
    M.nnz++;

    if (element.col == element.row)
      continue;

    element.row = col;
    element.col = row;
    M.elements[M.nnz] = element;
    M.nnz++;
  }

  qsort(M.elements, M.nnz, sizeof(matrix_entry), compare_matrix_elements);

  nnz = M.nnz;
  for (int j = 1; j < scale; j++)
  {
    for (int i = 0; i < nnz; i++)
    {
      matrix_entry element = M.elements[i];
      element.col = element.col + j*width;
      element.row = element.row + j*height;
      M.elements[M.nnz] = element;
      M.nnz++;
    }
  }

  *N = width*scale;

  return M;
}

// Generate a random, sparse, symmetric, positive-definite matrix
sparse_matrix generate_sparse_matrix(unsigned N, double percent_nonzero)
{
  sparse_matrix M = {0, NULL};
  unsigned allocated = 0;

  double *rowsum = calloc(N, sizeof(double));

  // Loop over rows
  for (unsigned y = 0; y < N; y++)
  {
    int stride;
    for (int x = y; x < N; x += stride)
    {
      // Calculate stride to next non-zero
      // +/- 1 to make things a little less predicatable
      stride = 100/percent_nonzero + (2*(rand() / (double)RAND_MAX) - 1);
      if (stride == 0)
        stride = 1;

      // Allocate more element data if needed
      if (M.nnz+2 > allocated)
      {
        allocated += 1024;
        M.elements = realloc(M.elements, allocated*sizeof(matrix_entry));
      }

      double value = rand() / (double)RAND_MAX;

      matrix_entry element;
      element.value = value;
      element.col   = x;
      element.row   = y;

      M.elements[M.nnz] = element;
      M.nnz++;

      rowsum[y] += value;

      if (x == y)
        continue;

      element.value = value;
      element.col   = y;
      element.row   = x;

      M.elements[M.nnz] = element;
      M.nnz++;

      rowsum[x] += value;
    }
  }

  // TODO: Get rid of this sort if we don't care about element order
  qsort(M.elements, M.nnz, sizeof(matrix_entry), compare_matrix_elements);

  for (unsigned i = 0; i < M.nnz; i++)
  {
    matrix_entry element = M.elements[i];

    // Increase the diagonal by the row-sum
    if (element.col == element.row)
    {
      element.value += rowsum[element.row];
    }

    M.elements[i] = element;
  }

  free(rowsum);

  return M;
}

// Generate a random, sparse, symmetric, positive-definite matrix
// TODO: Delete this once confirmed that faster (less random) approach is fine
sparse_matrix generate_sparse_matrix_slow(unsigned N, double percent_nonzero)
{
  sparse_matrix M = {0, NULL};
  unsigned allocated = 0;

  double *rowsum = calloc(N, sizeof(double));

  for (unsigned y = 0; y < N; y++)
  {
    for (unsigned x = y; x < N; x++)
    {
      // Decide if element should be non-zero
      // Always true for the diagonal
      double p = rand() / (double)RAND_MAX;
      if (p >= ((percent_nonzero/100) - 1.0/N) && x != y)
        continue;

      // Allocate more element data if needed
      if (M.nnz+2 > allocated)
      {
        allocated += 1024;
        M.elements = realloc(M.elements, allocated*sizeof(matrix_entry));
      }

      double value = rand() / (double)RAND_MAX;

      matrix_entry element;
      element.value = value;
      element.col   = x;
      element.row   = y;

      M.elements[M.nnz] = element;
      M.nnz++;

      rowsum[y] += value;

      if (x == y)
        continue;

      element.col   = y;
      element.row   = x;

      M.elements[M.nnz] = element;
      M.nnz++;

      rowsum[x] += value;
    }
  }

  qsort(M.elements, M.nnz, sizeof(matrix_entry), compare_matrix_elements);

  for (unsigned i = 0; i < M.nnz; i++)
  {
    matrix_entry element = M.elements[i];

    // Increase the diagonal by the row-sum
    if (element.col == element.row)
    {
      element.value += rowsum[element.row] + 1;
    }

    M.elements[i] = element;
  }

  free(rowsum);

  return M;
}
