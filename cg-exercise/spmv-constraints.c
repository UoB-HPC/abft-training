#include <stdio.h>
#include <stdlib.h>

#include "common.h"

// Initialize ECC for a sparse matrix
void init_matrix_ecc(sparse_matrix M)
{
  // Not using ECC - nothing to do
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

      // Skip last column
      if (i < end-1)
      {
        // Compare this column index to the next column index (if same row)
        uint32_t next_col = matrix.elements[i+1].col;
        if (element.col >= next_col)
        {
          printf("column index order violated for %d\n", i);
          exit(1);
        }
      }

      // Multiply element value by the corresponding vector value
      // and accumulate into row result
      tmp += element.value * vector[element.col];
    }

    // Store row total into result vector
    result[row] = tmp;
  }
}
