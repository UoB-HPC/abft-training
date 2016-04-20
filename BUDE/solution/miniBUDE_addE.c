//
// miniBUDE
//

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "crc_def.h"

#define FILE_LIGAND       "data/ligand.dat"
#define FILE_PROTEIN      "data/protein.dat"
#define FILE_FORCEFIELD   "data/forcefield.dat"
#define FILE_GEN0         "data/gen0.dat"

// Energy evaluation parameters
#define CNSTNT   45.0f
#define HBTYPE_F 70
#define HBTYPE_E 69
#define HARDNESS 38.0f
#define NPNPDIST  5.5f
#define NPPDIST   1.0f

// Search parameters
#define MUTATION_RATE   0.3f
#define GRID_SIZE      20

#ifndef M_PI_F
#define M_PI_F 3.141592653589793f
#endif

typedef struct
{
  float x, y, z;
  int type;
} Atom;

typedef struct
{
  int hbtype;
  float radius;
  float hphb;
  float elsc;
} FFParams;

struct
{
  int natlig;
  int natpro;
  int ntypes;

  int nposes;
  int nparents;
  int iterations;

  Atom *receptor;
  Atom *ligand;
  FFParams *forcefield;

  float *poses[6];

} params = { 0 };

//CRC32
CRC_32 initialCRC;
DataLength my_lengths;

double getTimestamp();
void loadParameters(int argc, char *argv[]);
void freeParameters();

void calculateEnergies(float *poses[6], float *energies);
void mutatePoses(float *poses[6]);
void sortPoses(float *poses[6], float *energies, int left, int right);

int main(int argc, char *argv[])
{

  //Final CRC check
  CRC_32 finalCRC;

  initialCRC.receptor = initialCRC.ligand = initialCRC.forcefield = 0;
  finalCRC.receptor = finalCRC.ligand = finalCRC.forcefield = 0;

  loadParameters(argc, argv);

//  for(int m = 0; m < params.natlig; m++)
//    printf("%i) my_type: %i\n",m, params.ligand[m].type);

  float *energies = malloc(params.nposes * sizeof(float));

  printf("\n");
  double start = getTimestamp();

  // First iteration (no mutation)
  calculateEnergies(params.poses, energies);
  sortPoses(params.poses, energies, 0, params.nposes - 1);
  printf("iteration %2d lowest energy = %7.2f\n", 0, energies[0]);


  // Modifying critical data to affect results.
  // Atoms 4, 5 & 16 have atom type different from 0
  int *my_type = &params.ligand[16].type;
  float *my_x  = &params.ligand[0].x;
  *my_type = (*my_type << 1);
  *my_x += .0000001f;

  // Run search
  for (int itr = 1; itr < params.iterations; itr++)
      {
    mutatePoses(params.poses);

    calculateEnergies(params.poses, energies);

    sortPoses(params.poses, energies, 0, params.nposes - 1);
    printf("iteration %2d lowest energy = %7.2f\n", itr, energies[0]);
  }

  // Checking CRC and the end of Docking.
  finalCRC.receptor = crc32(finalCRC.receptor, params.receptor,
      my_lengths.receptor);
  finalCRC.ligand = crc32(finalCRC.ligand, params.ligand, my_lengths.ligand);
  finalCRC.forcefield = crc32(finalCRC.forcefield, params.forcefield,
      my_lengths.forcefield);

  bool dataOK = isSameCRC(initialCRC, finalCRC);

  if (!dataOK) {
    printf("\nData corruption found:\nInitial CRC:\n");
    printCRC(initialCRC);
    printf("\nFinal CRC:\n");
    printCRC(finalCRC);
  }

  double end = getTimestamp();

  // Print top poses
  printf("\n  energy          rotation         translation\n");
  for (int i = 0; i < 5; i++)
      {
    printf(" %7.2f     %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f\n",
        energies[i],
        params.poses[0][i], params.poses[1][i], params.poses[2][i],
        params.poses[3][i], params.poses[4][i], params.poses[5][i]);
  }

  printf("\nTotal time: %7.2lf ms\n\n", (end - start) * 1e-3);

  free(energies);
  freeParameters();
}

void calculateEnergies(float *poses[6], float *energies)
{
#pragma omp parallel for
  for (unsigned i = 0; i < params.nposes; i++)
      {
    float etot = 0;

    // Compute transformation matrix
    const float sx = sin(poses[0][i]);
    const float cx = cos(poses[0][i]);
    const float sy = sin(poses[1][i]);
    const float cy = cos(poses[1][i]);
    const float sz = sin(poses[2][i]);
    const float cz = cos(poses[2][i]);

    float transform[3][4];
    transform[0][0] = cy * cz;
    transform[0][1] = sx * sy * cz - cx * sz;
    transform[0][2] = cx * sy * cz + sx * sz;
    transform[0][3] = poses[3][i];
    transform[1][0] = cy * sz;
    transform[1][1] = sx * sy * sz + cx * cz;
    transform[1][2] = cx * sy * sz - sx * cz;
    transform[1][3] = poses[4][i];
    transform[2][0] = -sy;
    transform[2][1] = sx * cy;
    transform[2][2] = cx * cy;
    transform[2][3] = poses[5][i];

    // Loop over ligand atoms
    int il = 0;
    do
    {
      // Load ligand atom data
      const Atom l_atom = params.ligand[il];
      const FFParams l_params = params.forcefield[l_atom.type];
      const int lhphb_ltz = l_params.hphb < 0.f;
      const int lhphb_gtz = l_params.hphb > 0.f;

      // Transform ligand atom
      float lpos_x = transform[0][3]
          + l_atom.x * transform[0][0]
          + l_atom.y * transform[0][1]
          + l_atom.z * transform[0][2];
      float lpos_y = transform[1][3]
          + l_atom.x * transform[1][0]
          + l_atom.y * transform[1][1]
          + l_atom.z * transform[1][2];
      float lpos_z = transform[2][3]
          + l_atom.x * transform[2][0]
          + l_atom.y * transform[2][1]
          + l_atom.z * transform[2][2];

      // Loop over protein atoms
      int ip = 0;
      do
      {
        // Load protein atom data
        const Atom p_atom = params.receptor[ip];
        const FFParams p_params = params.forcefield[p_atom.type];

        const float radij = p_params.radius + l_params.radius;
        const float r_radij = 1.f / radij;

        const float elcdst =
            (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F)
            ? 4.f : 2.f;
        const float elcdst1 =
            (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F)
            ? 0.25f : 0.5f;
        const int type_E =
            ((p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E));

        const int phphb_ltz = p_params.hphb < 0.f;
        const int phphb_gtz = p_params.hphb > 0.f;
        const int phphb_nz = p_params.hphb != 0.f;
        const float p_hphb =
            p_params.hphb * (phphb_ltz && lhphb_gtz ? -1.f : 1.f);
        const float l_hphb =
            l_params.hphb * (phphb_gtz && lhphb_ltz ? -1.f : 1.f);
        const float distdslv =
            (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST)
                         :
                         (lhphb_ltz ? NPPDIST : -FLT_MAX));
        const float r_distdslv = 1.f / distdslv;

        const float chrg_init = l_params.elsc * p_params.elsc;
        const float dslv_init = p_hphb + l_hphb;

        // Calculate distance between atoms
        const float x = lpos_x - p_atom.x;
        const float y = lpos_y - p_atom.y;
        const float z = lpos_z - p_atom.z;
        const float distij = sqrt(x * x + y * y + z * z);

        // Calculate the sum of the sphere radii
        const float distbb = distij - radij;
        const int zone1 = (distbb < 0.f);

        // Calculate steric energy
        etot += (1.f - (distij * r_radij)) * (zone1 ? 2 * HARDNESS : 0.f);

        // Calculate formal and dipole charge interactions
        float chrg_e = chrg_init
            * ((zone1 ? 1 : (1.f - distbb * elcdst1))
                * (distbb < elcdst ? 1 : 0.f));
        float neg_chrg_e = -fabs(chrg_e);
        chrg_e = type_E ? neg_chrg_e : chrg_e;
        etot += chrg_e * CNSTNT;

        // Calculate the two cases for Nonpolar-Polar repulsive interactions
        float coeff = (1.f - (distbb * r_distdslv));
        float dslv_e = dslv_init * ((distbb < distdslv && phphb_nz) ? 1 : 0.f);
        dslv_e *= (zone1 ? 1 : coeff);
        etot += dslv_e;

      } while (++ip < params.natpro); // loop over protein atoms
    } while (++il < params.natlig); // loop over ligand atoms

    // Write result
    energies[i] = etot * 0.5f;
  }
}

FILE* openFile(const char *name, long *length)
{
  FILE *file = NULL;
  if (!(file = fopen(name, "rb")))
  {
    fprintf(stderr, "Failed to open '%s'\n", name);
    exit(1);
  }

  fseek(file, 0, SEEK_END);
  *length = ftell(file);
  rewind(file);

  return file;
}

int parseInt(const char *str)
{
  char *next;
  int value = strtoul(str, &next, 10);
  return strlen(next) ? -1 : value;
}

void loadParameters(int argc, char *argv[])
{
  // Defaults
  params.iterations = 8;
  params.nposes = 4096;
  params.nparents = 128;

  for (int i = 1; i < argc; i++)
      {
    if (!strcmp(argv[i], "--iterations") || !strcmp(argv[i], "-i"))
        {
      if (++i >= argc || (params.iterations = parseInt(argv[i])) < 0)
          {
        printf("Invalid number of iterations\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--numposes") || !strcmp(argv[i], "-n"))
        {
      if (++i >= argc || (params.nposes = parseInt(argv[i])) < 0)
          {
        printf("Invalid number of poses\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--numparents") || !strcmp(argv[i], "-p"))
        {
      if (++i >= argc || (params.nparents = parseInt(argv[i])) < 1)
          {
        printf("Invalid number of parents\n");
        exit(1);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
      printf("\n");
      printf("Usage: ./bude-benchmark [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h  --help               Print this message\n");
      printf("  -i  --iterations I       Run search for I iterations\n");
      printf("  -n  --numposes   N       Generate N poses per iteration\n");
      printf("  -p  --numparents P       Use P parents per iteration\n");
      printf("\n");
      exit(0);
    }
    else
    {
      printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
      exit(1);
    }
  }

  FILE *file = NULL;
  long length;

  file = openFile(FILE_LIGAND, &length);
  params.natlig = length / sizeof(Atom);
  params.ligand = malloc(params.natlig * sizeof(Atom));
  fread(params.ligand, sizeof(Atom), params.natlig, file);
  fclose(file);

  //Get CRC for ligand
  my_lengths.ligand = length;
  initialCRC.ligand = crc32(initialCRC.ligand, params.ligand, length);

  file = openFile(FILE_PROTEIN, &length);
  params.natpro = length / sizeof(Atom);
  params.receptor = malloc(params.natpro * sizeof(Atom));
  fread(params.receptor, sizeof(Atom), params.natpro, file);
  fclose(file);

  //Get CRC for receptor
  my_lengths.receptor = length;
  initialCRC.receptor = crc32(initialCRC.receptor, params.receptor, length);

  file = openFile(FILE_FORCEFIELD, &length);
  params.ntypes = length / sizeof(FFParams);
  params.forcefield = malloc(params.ntypes * sizeof(FFParams));
  fread(params.forcefield, sizeof(FFParams), params.ntypes, file);
  fclose(file);

  //Get CRC for forcefield.
  my_lengths.forcefield = length;
  initialCRC.forcefield = crc32(initialCRC.forcefield, params.forcefield,
      length);

//  printCRC(initialCRC);

  file = openFile(FILE_GEN0, &length);
  for (int i = 0; i < 6; i++)
    params.poses[i] = malloc(params.nposes * sizeof(float));

  // Repeat poses from GEN0 file to make up correct number of poses
  long available = length / 6 / sizeof(float);
  int nposes = 0;
  while (nposes < params.nposes)
  {
    long fetch = params.nposes - nposes;
    if (fetch > available)
      fetch = available;

    for (int i = 0; i < 6; i++)
        {
      fseek(file, i * available * sizeof(float), SEEK_SET);
      fread(params.poses[i] + nposes, sizeof(float), fetch, file);
    }
    rewind(file);

    nposes += fetch;
  }
  fclose(file);
}

void freeParameters()
{
  free(params.ligand);
  free(params.receptor);
  free(params.forcefield);
  for (int i = 0; i < 6; i++)
    free(params.poses[i]);
}

double getTimestamp()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec * 1e6;
}

// Simple pose mutation that takes the best NUM_PARENTS poses and generates
// an equal amount of children from each, using destructive mutation.
// Assumes input poses are sorted by energy value (ascending).
void mutatePoses(float *poses[6])
{
  // Allocate temporary arrays for child poses
  float *newPoses[6];
  for (int i = 0; i < 6; i++)
    newPoses[i] = malloc(params.nposes * sizeof(float));

  int childrenPerParent = ceil(params.nposes / (float) params.nparents);
  for (int parent = 0; parent < params.nparents; parent++)
      {
    // Generate children for this parent
    for (int i = 0; i < childrenPerParent; i++)
        {
      int child = parent * childrenPerParent + i;
      if (child >= params.nposes)
        break;

      for (int p = 0; p < 6; p++)
          {
        newPoses[p][child] = poses[p][parent];

        // First child is exact copy of parent
        if (i == 0)
          continue;

        // Decide whether to mutate this component
        float r = rand() / (float) RAND_MAX;
        if (r < MUTATION_RATE)
        {
          float v = rand() / (float) RAND_MAX;
          if (p < 3)
              {
            // Pick a random angle
            newPoses[p][child] = v * (2 * M_PI_F) - M_PI_F;
          }
          else
          {
            // Pick a random translation
            newPoses[p][child] = v * GRID_SIZE - (GRID_SIZE / 2);
          }
        }
      }
    }
  }

  // Copy new poses to output
  for (int i = 0; i < 6; i++)
      {
    memcpy(poses[i], newPoses[i], params.nposes * sizeof(float));
    free(newPoses[i]);
  }
}

// Simple in-place quick-sort that sorts both the poses and energies
// based on energy value (ascending)
void sortPoses(float *poses[6], float *energies, int left, int right)
{
  if (right - left < 1)
    return;

  // Select the first pose as the pivot
  float pivot = energies[left];

  // Parition poses based on pivot
  int l = left;
  int r = right;
  while (l <= r)
  {
    while (energies[l] < pivot)
      l++;
    while (energies[r] > pivot)
      r--;

    if (l <= r)
        {
      float tmp;

      // Swap energies
      tmp = energies[l];
      energies[l] = energies[r];
      energies[r] = tmp;

      // Swap poses
      for (int i = 0; i < 6; i++)
          {
        tmp = poses[i][l];
        poses[i][l] = poses[i][r];
        poses[i][r] = tmp;
      }

      l++;
      r--;
    }
  }

  // Sort partitions
  sortPoses(poses, energies, left, r);
  sortPoses(poses, energies, l, right);
}
