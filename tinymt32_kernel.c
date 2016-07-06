#ifndef TINYMT32_KERNEL_CUH
#define TINYMT32_KERNEL_CUH
#include <assert.h>
/**
 * @file tinymt32_kernel.cuh
 *
 * @brief CUDA implementation of TinyMT32.
 *
 * This is CUDA implementation of TinyMT32 pseudo-random number generator.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2011 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see LICENSE.txt
 */
#include <cuda.h>
#include <stdint.h>
#include <errno.h>
/* =====================================
   DEFINITIONS FOR USERS
   ===================================== */
/**
 * TinyMT structure
 * mat1, mat2, tmat must be set from tinymt32dc output before initialize
 */
typedef struct TINYMT32_STATUS_T {
    uint32_t status[4];
    uint32_t mat1;
    uint32_t mat2;
    uint32_t tmat;
} tinymt32_status_t;

/**
 * Initialize TinyMT structure by seed and parameters.
 *
 * This function must be called before tinymt32_uint32(),
 * tinymt32_single(), tinymt32_single12().
 * mat1, mat2, tmat in tinymt must be set before this call.
 *
 * @param tinymt TinyMT structure to be initialized.
 * @param seed seed of randomness.
 */
__device__ void tinymt32_init(tinymt32_status_t* tinymt, uint32_t seed);

/**
 * Generate 32bit unsigned integer r (0 <= r < 2<sup>32</sup>)
 *
 * @param tinymt TinyMT structure
 * @return 32bit unsigned integer
 */
__device__ uint32_t tinymt32_uint32(tinymt32_status_t * tinymt);

/**
 * Generate single precision floating point number r (0.0 <= r < 1.0)
 *
 * @param tinymt TinyMT structure
 * @return single precision floating point number
 */
__device__ float tinymt32_single(tinymt32_status_t * tinymt);

/**
 * Generate single precision floating point number r (1.0 <= r < 2.0).
 *
 * This function may little bit faster than tinymt32_single().
 *
 * @param tinymt TinyMT structure
 * @return single precision floating point number
 */
__device__ float tinymt32_single12(tinymt32_status_t * tinymt);

/* =====================================
   DEFINITIONS FOR INTERNAL USE
   ===================================== */
#define TINYMT32_SHIFT0 1
#define TINYMT32_SHIFT1 10
#define TINYMT32_MIN_LOOP 8
#define TINYMT32_PRE_LOOP 8
#define TINYMT32_MASK 0x7fffffffU
#define TINYMT32_SINGLE_MASK 0x3f800000U

__device__ void tinymt32_next(tinymt32_status_t * tinymt);
__device__ uint32_t tinymt32_temper(tinymt32_status_t * tinymt);



/* =====================================
   FUNCTIONS
   ===================================== */
/**
 * The state transition function.
 * @param tinymt the internal state.
 */
__device__ void tinymt32_next(tinymt32_status_t * tinymt)
{
    uint32_t y = tinymt->status[3];
    uint32_t x = (tinymt->status[0] & TINYMT32_MASK)
	^ tinymt->status[1] ^ tinymt->status[2];
    x ^= (x << TINYMT32_SHIFT0);
    y ^= (y >> TINYMT32_SHIFT0) ^ x;
    tinymt->status[0] = tinymt->status[1];
    tinymt->status[1] = tinymt->status[2];
    tinymt->status[2] = x ^ (y << TINYMT32_SHIFT1);
    tinymt->status[3] = y;
    if (y & 1) {
	tinymt->status[1] ^= tinymt->mat1;
	tinymt->status[2] ^= tinymt->mat2;
    }
}

/**
 * The tempering function.
 *
 * This function improves the equidistribution property of
 * outputs.
 * @param tinymt the internal state.
 * @return tempered output
 */
__device__ uint32_t tinymt32_temper(tinymt32_status_t * tinymt)
{
    uint32_t t0, t1;
    t0 = tinymt->status[3];
    t1 = tinymt->status[0] + (tinymt->status[2] >> 8);
    t0 ^= t1;
    if (t1 & 1) {
	t0 ^= tinymt->tmat;
    }
    return t0;
}

__device__ uint32_t tinymt32_uint32(tinymt32_status_t * tinymt)
{
    tinymt32_next(tinymt);
    return tinymt32_temper(tinymt);
}

__device__ float tinymt32_single12(tinymt32_status_t * tinymt)
{
    uint32_t t0;
    tinymt32_next(tinymt);
    t0 = tinymt32_temper(tinymt);
    t0 = t0 >> 9;
    t0 ^= TINYMT32_SINGLE_MASK;
    return __int_as_float(t0);
}

__device__ float tinymt32_single(tinymt32_status_t * tinymt)
{
    return tinymt32_single12(tinymt) - 1.0f;
}

__device__ void tinymt32_init(tinymt32_status_t* tinymt, uint32_t seed) {
    tinymt->status[0] = seed;
    tinymt->status[1] = tinymt->mat1;
    tinymt->status[2] = tinymt->mat2;
    tinymt->status[3] = tinymt->tmat;
    for (int i = 1; i < TINYMT32_MIN_LOOP; i++) {
	tinymt->status[i & 3] ^= i + 1812433253U *
	    (tinymt->status[(i - 1) & 3]
	     ^ (tinymt->status[(i - 1) & 3] >> 30));
    }
    if ((tinymt->status[0] & TINYMT32_MASK) == 0 &&
	tinymt->status[1] == 0 &&
	tinymt->status[2] == 0 &&
	tinymt->status[3] == 0) {
	tinymt->status[0] = 'T';
	tinymt->status[1] = 'I';
	tinymt->status[2] = 'N';
	tinymt->status[3] = 'Y';
    }
    for (int i = 0; i < TINYMT32_PRE_LOOP; i++) {
	tinymt32_next(tinymt);
    }
}


/* initializes mt[N] with a seed */
static int read_line32(uint32_t *mat1, uint32_t *mat2, uint32_t *tmat, FILE *fp)
{
#define BUFF_SIZE 500
    char buff[BUFF_SIZE];
    char * p;
    uint32_t num;
    errno = 0;
    for (;;) {
	if (feof(fp) || ferror(fp)) {
	    return -1;
	}
	fgets(buff, BUFF_SIZE, fp);
	if (errno) {
	    return errno;
	}
	if (buff[0] != '#') {
	    break;
	}
    }
    p = buff;
    for (int i = 0; i < 3; i++) {
	p = strchr(p, ',');
	if (p == NULL) {
	    return -1;
	}
	p++;
    }
    num = strtoul(p, &p, 16);
    if (errno) {
	return errno;
    }
    *mat1 = num;
    p++;
    num = strtoul(p, &p, 16);
    if (errno) {
	return errno;
    }
    *mat2 = num;
    p++;
    num = strtoul(p, &p, 16);
    if (errno) {
	return errno;
    }
    *tmat = num;
    return 0;
}

int tinymt32_set_params(const char * filename,
			uint32_t * params,
			int num_param)
{
    FILE *ifp;
    int rc;
    uint32_t mat1 = 0;
    uint32_t mat2 = 0;
    uint32_t tmat = 0;
    int i;
    ifp = fopen(filename, "r");
    if (ifp == NULL) {
	return -1;
    }
    for (i = 0; i < num_param; i++) {
	rc = read_line32(&mat1, &mat2, &tmat, ifp);
	if (rc != 0) {
	    return -2;
	}
	params[i * 3 + 0] = mat1;
	params[i * 3 + 1] = mat2;
	params[i * 3 + 2] = tmat;
    }
    fclose(ifp);
    return 0;
}



__global__ void init_genrand_kernel(tinymt32_status_t *mt, uint32_t *seed, uint32_t *params)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  tinymt32_status_t * tinymt = mt + tid;
  uint32_t * s = seed + tid;
  
  tinymt->mat1 = params[3*tid];
  tinymt->mat2 = params[3*tid+1];
  tinymt->tmat = params[3*tid+2];
  
  tinymt32_init(tinymt, *s); 
}

void init_genrand_dev(const char *filename, unsigned int *s, int NB, int NBT, tinymt32_status_t **pmt)
{
  uint32_t *seed, *paramsdev, *params;
  int NT = NB*NBT;

  params = (uint32_t *) malloc(3*NT*sizeof(uint32_t));
  
  HANDLE_ERROR( cudaMalloc((void**)   pmt,  NT*sizeof(tinymt32_status_t)) );
  HANDLE_ERROR( cudaMalloc((void**) &seed,  NT*sizeof(uint32_t         )) );
  HANDLE_ERROR( cudaMalloc((void**) &paramsdev,  3*NT*sizeof(uint32_t  )) );

  assert( tinymt32_set_params(filename, params, NT) == 0);

  HANDLE_ERROR( cudaMemcpy(seed, s, NT*sizeof(uint32_t), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(paramsdev, params, 3*NT*sizeof(uint32_t), cudaMemcpyHostToDevice) );

  init_genrand_kernel<<<NB, NBT>>>(*pmt, seed, paramsdev);
  
  HANDLE_ERROR( cudaFree(seed) );
  HANDLE_ERROR( cudaFree(paramsdev) );
  free(params);
}



#undef TINYMT32_SHIFT0
#undef TINYMT32_SHIFT1
#undef TINYMT32_MIN_LOOP
#undef TINYMT32_PRE_LOOP
#undef TINYMT32_MASK
#undef TINYMT32_SINGLE_MASK

#endif
