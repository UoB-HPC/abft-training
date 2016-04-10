//
// Simple conjugate gradient solver
//

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

struct
{
  int    n;              // order of matrix
  int    max_itrs;       // max iterations to run
  double percent_nzero;  // percent of matrix to be non-zero
  double conv_threshold; // convergence threshold to stop CG

  int    inject_bitflip; // flip a random bit in the matrix
} params;

// TODO: Use same layout as slides (col,row,value)
typedef struct
{
  uint32_t col;
  uint32_t row;
  double value;
} matrix_entry;

typedef struct
{
  unsigned nnz;
  matrix_entry *elements;
} sparse_matrix;

sparse_matrix generate_sparse_matrix(unsigned N, double percent_nonzero);

double   get_timestamp();
void     parse_arguments(int argc, char *argv[]);

uint32_t ecc_compute_high8(matrix_entry element);
void     ecc_correct_flip(matrix_entry *element, uint32_t syndrome);

// Sparse matrix vector product
// Multiplies `matrix` by `vector` and stores answer in `result`
// The matrix and vector dimensions are `N`
void spmv(sparse_matrix matrix, double *vector, double *result, unsigned N)
{
  // Initialize result vector to zero
  for (unsigned i = 0; i < N; i++)
    result[i] = 0.0;

  // Loop over non-zeros in matrix
  for (unsigned i = 0; i < matrix.nnz; i++)
  {
    // Load non-zero element
    matrix_entry element = matrix.elements[i];

    // Check ECC
    uint32_t syndrome = ecc_compute_high8(element);
    if (syndrome)
    {
     ecc_correct_flip(&element, syndrome);
     matrix.elements[i] = element;
    }

    // Mask out ECC from high order column bits
    element.col &= 0x00FFFFFF;

    // Multiply element value by the corresponding vector value
    // and accumulate into result vector
    result[element.col] += element.value * vector[element.row];
  }
}

int main(int argc, char *argv[])
{
  parse_arguments(argc, argv);

  sparse_matrix A = generate_sparse_matrix(params.n, params.percent_nzero);

  // Add ECC protection to matrix elements
  for (unsigned i = 0; i < A.nnz; i++)
  {
    matrix_entry element = A.elements[i];

    // Generate ECC and store in high order column bits
    element.col |= ecc_compute_high8(element);

    A.elements[i] = element;
  }

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
    int bit   = rand() % 128;
    int word  = bit / 32;
    printf("*** flipping bit %d of (%d,%d) ***\n", bit,
           A.elements[index].col & 0x00FFFFFF,
           A.elements[index].row & 0x00FFFFFF);

    ((uint32_t*)(A.elements+index))[word] ^= 1<<bit;
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

    if (itr % 5 == 0)
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
    return _a.col < _b.col;
  }
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

#define ECC7_P1_0 0x80AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0xAAAAAAAA
#define ECC7_P1_3 0x55AAAAAA

#define ECC7_P2_0 0x4033366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0xCCCCCCCC
#define ECC7_P2_3 0x66CCCCCC

#define ECC7_P3_0 0x20C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0xF0F0F0F0
#define ECC7_P3_3 0x78F0F0F0

#define ECC7_P4_0 0x10FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x00FF00FF
#define ECC7_P4_3 0x80FF00FF

#define ECC7_P5_0 0x08FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x00FFFF00
#define ECC7_P5_3 0x00FFFF00

#define ECC7_P6_0 0x04000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0xFF000000
#define ECC7_P6_3 0x00FFFFFF

#define ECC7_P7_0 0x02000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0xFFFFFFFF
#define ECC7_P7_3 0x00FFFFFF

uint32_t ecc_compute_high8(matrix_entry element)
{
  uint32_t *data = (uint32_t*)&element;

  uint32_t result = 0;

  uint32_t p;

  p = (data[0] & ECC7_P1_0) ^ (data[1] & ECC7_P1_1) ^
      (data[2] & ECC7_P1_2) ^ (data[3] & ECC7_P1_3);
  result |= __builtin_parity(p) << 31;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^
      (data[2] & ECC7_P2_2) ^ (data[3] & ECC7_P2_3);
  result |= __builtin_parity(p) << 30;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^
      (data[2] & ECC7_P3_2) ^ (data[3] & ECC7_P3_3);
  result |= __builtin_parity(p) << 29;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^
      (data[2] & ECC7_P4_2) ^ (data[3] & ECC7_P4_3);
  result |= __builtin_parity(p) << 28;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^
      (data[2] & ECC7_P5_2) ^ (data[3] & ECC7_P5_3);
  result |= __builtin_parity(p) << 27;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^
      (data[2] & ECC7_P6_2) ^ (data[3] & ECC7_P6_3);
  result |= __builtin_parity(p) << 26;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^
      (data[2] & ECC7_P7_2) ^ (data[3] & ECC7_P7_3);
  result |= __builtin_parity(p) << 25;

  return result;
}

static int is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

void ecc_correct_flip(matrix_entry *element, uint32_t syndrome)
{
  uint32_t *data = (uint32_t*)element;

  // Compute position of flipped bit
  uint32_t hamm_bit = 0;
  for (int p = 1; p <= 7; p++)
  {
    if ((syndrome >> (32-p)) & 0x1)
      hamm_bit += 0x1<<(p-1);
  }

  // Map to actual data bit position
  uint32_t data_bit = hamm_bit - (32-__builtin_clz(hamm_bit)) - 1;
  if (is_power_of_2(hamm_bit))
    data_bit = __builtin_clz(hamm_bit);

  // Unflip bit
  uint32_t word = data_bit / 32;
  data[word] ^= 0x1 << (data_bit % 32);

  printf("[ECC] corrected bit %u of (%d,%d)\n",
         data_bit, element->col & 0x00FFFFFF, element->row & 0x00FFFFFF);
}

void gen_ecc7_masks()
{
  for (uint32_t p = 1; p <= 7; p++)
  {
    uint32_t x = 3;
    for (int w = 0; w < 4; w++)
    {
      uint32_t mask = 0;
      for (uint32_t b = 0; b < 32; b++)
      {
        if (is_power_of_2(x))
          x++;

        uint32_t bit = w*32 + b;
        if (bit >= (32-7) && bit < 32)
        {
          if ((32-bit) == p)
            mask |= 0x1 << b;
        }
        else if (x & (0x1<<(p-1)))
          mask |= 0x1 << b;

        x++;
      }
      if (w == 0)
        mask &= 0xFEFFFFFF;
      printf("#define ECC7_P%d_%d 0x%08X\n", p, w, mask);
    }
    printf("\n");
  }
}
