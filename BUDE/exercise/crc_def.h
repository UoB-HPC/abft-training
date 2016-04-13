
#ifndef CRC_DEF_H_
#define CRC_DEF_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
  uint32_t receptor;
  uint32_t ligand;
  uint32_t forcefield;
} CRC_32;

typedef struct {
  unsigned long receptor;
  unsigned long ligand;
  unsigned long forcefield;
} DataLength;

uint32_t crc32(uint32_t crc, const void *buf, size_t size);
void printCRC(CRC_32 values);
uint32_t calculateCRC( void const * pData, const size_t length);

bool isSameCRC(CRC_32 initial, CRC_32 final);

#endif /* CRC_DEF_H_ */
