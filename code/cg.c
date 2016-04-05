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
