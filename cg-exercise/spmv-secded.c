#include <stdio.h>
#include <stdlib.h>

#include "common.h"

// Initialize ECC for a sparse matrix
void init_matrix_ecc(sparse_matrix M)
{
  // Add ECC protection to matrix elements
  for (unsigned i = 0; i < M.nnz; i++)
  {
    matrix_entry element = M.elements[i];

    // Generate ECC and store in high order column bits
    element.col |= ecc_compute_col8(element);

    // Compute overall parity bit for whole codeword
    element.col |= ecc_compute_overall_parity(element) << 24;

    M.elements[i] = element;
  }
}

// Sparse matrix vector product
// Multiplies `matrix` by `vector` and stores answer in `result`
// The matrix and vector dimensions are `N`
void spmv(sparse_matrix matrix, double *vector, double *result, unsigned N)
{
  // Loop over rows
  for (unsigned row = 0; row < N; row++)
  {
    double tmp = 0.0;

    // Loop over columns in this row
    uint32_t start = matrix.row_indices[row];
    uint32_t end   = matrix.row_indices[row+1];
    for (int i = start; i < end; i++)
    {
      // Load non-zero element
      matrix_entry element = matrix.elements[i];

      // Check parity bits
      uint32_t overall_parity = ecc_compute_overall_parity(element);
      uint32_t syndrome = ecc_compute_col8(element);
      if (overall_parity)
      {
        if (syndrome)
        {
          ecc_correct_col8(&element, syndrome);
        }
        else
        {
          // Correct overall parity bit
          printf("[ECC] corrected overall parity bit for %d\n", i);
          element.col ^= 0x1 << 24;
        }
        matrix.elements[i] = element;
      }
      else
      {
        if (syndrome)
        {
          // Overall parity fine but error in syndrom
          // Must be double-bit error - cannot correct this
          printf("[ECC] Double-bit error detected\n");
          exit(1);
        }
      }

      // Mask out ECC from high order column bits
      element.col &= 0x00FFFFFF;

      // Multiply element value by the corresponding vector value
      // and accumulate into row result
      tmp += element.value * vector[element.col];
    }

    // Store row total into result vector
    result[row] = tmp;
  }
}
