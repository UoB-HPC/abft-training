//
// Simple conjugate gradient solver
//

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct
{
  int    n;              // order of matrix
  int    max_itrs;       // max iterations to run
  double percent_nzero;  // percent of matrix to be non-zero
  double conv_threshold; // convergence threshold to stop CG
} params;

typedef struct
{
  double value;
  uint32_t row;
  uint32_t col;
} matrix_entry;

void     parseArguments(int argc, char *argv[]);

uint32_t ecc_compute_high8(matrix_entry element);
void     ecc_correct_flip(matrix_entry *element, uint32_t syndrome);

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

    // Check ECC
    uint32_t syndrome = ecc_compute_high8(element);
    if (syndrome)
    {
      ecc_correct_flip(&element, syndrome);
      matrix[i] = element;
    }

    // Mask out ECC from high order column bits
    element.col &= 0x00FFFFFF;

    output[element.col] += element.value * vector[element.row];
  }
}

int main(int argc, char *argv[])
{
  parseArguments(argc, argv);

  matrix_entry *A = NULL;

  double *b = malloc(params.n*sizeof(double));
  double *x = malloc(params.n*sizeof(double));
  double *r = malloc(params.n*sizeof(double));
  double *p = malloc(params.n*sizeof(double));
  double *w = malloc(params.n*sizeof(double));

  // Initialize symmetric sparse matrix A and vectors b and x
  unsigned nnz = 0;
  unsigned allocated = 0;
  for (unsigned y = 0; y < params.n; y++)
  {
    for (unsigned x = y; x < params.n; x++)
    {
      // Decide if element should be non-zero
      // Always true for the diagonal
      double p = rand() / (double)RAND_MAX;
      if (p >= ((params.percent_nzero/100) - 1.0/params.n) && x != y)
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

      // Generate ECC and store in high order column bits
      element.col |= ecc_compute_high8(element);

      A[nnz] = element;
      nnz++;

      if (x == y)
        continue;

      element.value = value;
      element.row   = x;
      element.col   = y;

      // Generate ECC and store in high order column bits
      element.col |= ecc_compute_high8(element);

      A[nnz] = element;
      nnz++;
    }

    b[y] = rand() / (double)RAND_MAX;
    x[y] = 0.0;
  }

  printf("\n");
  printf("matrix size           = %u x %u\n", params.n, params.n);
  printf("number of non-zeros   = %u (%.2lf%%)\n",
         nnz, nnz/(double)(params.n*params.n)*100);
  printf("maximum iterations    = %u\n", params.max_itrs);
  printf("convergence threshold = %g\n", params.conv_threshold);
  printf("\n");

  // r = b - Ax;
  // p = r
  spmv(A, x, r, params.n, nnz);
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
    spmv(A, p, w, params.n, nnz);

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

    if (itr % 100 == 0)
      printf("iteration %5u :  rr = %12.4lf\n", itr, rr);
  }

  printf("\n");
  printf("ran for %u iterations\n", itr);

  // Compute Ax
  double *Ax = malloc(params.n*sizeof(double));
  spmv(A, x, Ax, params.n, nnz);

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

  free(A);
  free(b);
  free(x);
  free(r);
  free(p);
  free(w);
  free(Ax);

  return 0;
}

double parseDouble(const char *str)
{
  char *next;
  double value = strtod(str, &next);
  return strlen(next) ? -1 : value;
}

int parseInt(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void parseArguments(int argc, char *argv[])
{
  // Set defaults
  params.n              = 1024;
  params.max_itrs       = 5000;
  params.percent_nzero  = 1.0;
  params.conv_threshold = 0.00001;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--convergence") || !strcmp(argv[i], "-c"))
    {
      if (++i >= argc || (params.conv_threshold = parseDouble(argv[i])) < 0)
      {
        printf("Invalid convergence threshold\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
    {
      if (++i >= argc || (params.max_itrs = parseInt(argv[i])) < 0)
      {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--norder") || !strcmp(argv[i], "-n"))
    {
      if (++i >= argc || (params.n = parseInt(argv[i])) < 1)
      {
        printf("Invalid matrix order\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--percent-nonzero") || !strcmp(argv[i], "-p"))
    {
      if (++i >= argc || (params.percent_nzero = parseDouble(argv[i])) < 0)
      {
        printf("Invalid number of parents\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./cg [OPTIONS]\n\n");
      printf("Options:\n");
      printf(
        "  -h  --help                Print this message\n"
        "  -c  --convergence    C    Convergence threshold\n"
        "  -i  --iterations     I    Maximum number of iterations\n"
        "  -n  --norder         N    Order of matrix A\n"
        "  -p  --percent-nzero  P    Percentage of A to be non-zero (approx)\n"
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

#define ECC7_P1_0 0x56AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0xAAAAAAAA
#define ECC7_P1_3 0x80AAAAAA

#define ECC7_P2_0 0x9B33366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0xCCCCCCCC
#define ECC7_P2_3 0x40CCCCCC

#define ECC7_P3_0 0xE3C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0xF0F0F0F0
#define ECC7_P3_3 0x20F0F0F0

#define ECC7_P4_0 0x03FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x00FF00FF
#define ECC7_P4_3 0x10FF00FF

#define ECC7_P5_0 0x03FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x00FFFF00
#define ECC7_P5_3 0x08FFFF00

#define ECC7_P6_0 0xFC000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0xFF000000
#define ECC7_P6_3 0x04FFFFFF

#define ECC7_P7_0 0x00000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0xFFFFFFFF
#define ECC7_P7_3 0x02FFFFFF

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
    data_bit = __builtin_clz(hamm_bit) + 96;

  printf("[ECC] correcting bit %u\n", data_bit);

  // Unflip bit
  uint32_t word = data_bit / 32;
  data[word] ^= 0x1 << (data_bit % 32);
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
        if (bit >= (128-7))
        {
          if ((128-bit) == p)
            mask |= 0x1 << b;
        }
        else if (x & (0x1<<(p-1)))
          mask |= 0x1 << b;

        x++;
      }
      if (w == 3)
        mask &= 0xFEFFFFFF;
      printf("#define ECC7_P%d_%d 0x%08X\n", p, w, mask);
    }
    printf("\n");
  }
}
