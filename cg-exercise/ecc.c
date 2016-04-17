#include <stdio.h>

#include "common.h"

#define ECC7_P1_0 0x56AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0x80AAAAAA

#define ECC7_P2_0 0x9B33366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0x40CCCCCC

#define ECC7_P3_0 0xE3C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0x20F0F0F0

#define ECC7_P4_0 0x03FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x10FF00FF

#define ECC7_P5_0 0x03FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x08FFFF00

#define ECC7_P6_0 0xFC000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0x04000000

#define ECC7_P7_0 0x00000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0x02FFFFFF

uint32_t ecc_compute_col8(matrix_entry element)
{
  uint32_t *data = (uint32_t*)&element;

  uint32_t result = 0;

  uint32_t p;

  p = (data[0] & ECC7_P1_0) ^ (data[1] & ECC7_P1_1) ^ (data[2] & ECC7_P1_2);
  result |= __builtin_parity(p) << 31;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^ (data[2] & ECC7_P2_2);
  result |= __builtin_parity(p) << 30;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^ (data[2] & ECC7_P3_2);
  result |= __builtin_parity(p) << 29;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^ (data[2] & ECC7_P4_2);
  result |= __builtin_parity(p) << 28;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^ (data[2] & ECC7_P5_2);
  result |= __builtin_parity(p) << 27;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^ (data[2] & ECC7_P6_2);
  result |= __builtin_parity(p) << 26;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^ (data[2] & ECC7_P7_2);
  result |= __builtin_parity(p) << 25;

  return result;
}

static int is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

uint32_t ecc_compute_overall_parity(matrix_entry element)
{
  uint32_t *data = (uint32_t*)&element;
  return __builtin_parity(data[0] ^ data[1] ^ data[2]);
}

void ecc_correct_col8(matrix_entry *element, uint32_t syndrome)
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
    data_bit = __builtin_clz(hamm_bit) + 64;

  // Unflip bit
  uint32_t word = data_bit / 32;
  data[word] ^= 0x1 << (data_bit % 32);

  printf("[ECC] corrected bit %u of (%d,%d)\n",
         data_bit, element->col & 0x00FFFFFF, 0); // TODO
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
        if (bit >= (32-8) && bit < 32)
        {
          if ((32-bit) == p)
            mask |= 0x1 << b;
        }
        else
        {
          if (x & (0x1<<(p-1)))
          {
            mask |= 0x1 << b;
          }
          x++;
        }
      }
      printf("#define ECC7_P%d_%d 0x%08X\n", p, w, mask);
    }
    printf("\n");
  }
}
