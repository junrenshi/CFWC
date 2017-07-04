#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda.h>
#include "cfwc.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
void HandleError( cudaError_t err,
  const char *file,
  int line );

#include "tinymt32_kernel.cuh"

__constant__ float XXdev[NMAX], YYdev[NMAX];
__constant__ float c1, c2x, c2y;
__constant__ float PP0dev[3*NMAX];

#define NRS  50

__constant__ float U0dev[NRS], U6dev[NRS], U12dev[NRS];
__constant__ float Rmaxdev;

#define NQMAX 6
__constant__ float qnr[NQMAX] = { 4.6797127951841172e-1f, -2.0279163407481332e-3f,
 -3.8081294567494075e-8f,  3.098876062284498e-15f,
 1.092766934009448e-24f, -1.669866367469883e-36f};
 __constant__ float qni[NQMAX] = { 1.9384005077761673e-1f, -8.3999045169589555e-4f,
   -1.5773788682580913e-8f,  1.283596493111571e-15f,
   4.526388845795783e-25f, -6.916812967567200e-37f};

   __device__ float normtheta1p_dev(float zr, float zi)
   {
    float sinnzr, sinnzi, cosnzr, cosnzi, sin2zr, sin2zi, cos2zr, cos2zi;
    float sr, si, cr, ci, thr, thi, d;
    
    sincosf(zr, &sr, &cr);
    si = sinhf(zi);
    ci = sqrtf(1.0f+si*si);
    
    sinnzr =  ci * sr;
    sinnzi =  si * cr;
    cosnzr =  ci * cr;
    cosnzi = -si * sr;
    
    cos2zr = -cosnzr*cosnzr + cosnzi*cosnzi + sinnzr*sinnzr - sinnzi*sinnzi;
    cos2zi = -2.0f * (cosnzr * cosnzi - sinnzr * sinnzi);
    sin2zr = -2.0f * (sinnzr * cosnzr - sinnzi * cosnzi);
    sin2zi = -2.0f * (sinnzr * cosnzi + sinnzi * cosnzr);

    thr = 0.0;
    thi = 0.0;
    for(int n = 0; n < NQMAX; ++n){
      thr += qnr[n] * sinnzr - qni[n] * sinnzi;
      thi += qnr[n] * sinnzi + qni[n] * sinnzr;

      sr = sinnzr;
      si = sinnzi;
      cr = cosnzr;
      ci = cosnzi;

      sinnzr =   cos2zr*sr - cos2zi*si + sin2zr*cr - sin2zi*ci;
      sinnzi =   cos2zi*sr + cos2zr*si + sin2zi*cr + sin2zr*ci;
      cosnzr = - sin2zr*sr + sin2zi*si + cos2zr*cr - cos2zi*ci;
      cosnzi = - sin2zi*sr - sin2zr*si + cos2zi*cr + cos2zr*ci;
    }

    d = zr*zr + zi*zi;
    
    return (thr*thr + thi*thi)/d;
  }


  inline __device__ float Q_rsqrt( float number )
  {
   unsigned int i;
   float x2, y;
   const float threehalfs = 1.5F;

   x2 = number * 0.5F;
   y  = number;
	i  = * (unsigned int * ) &y;                // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}


__device__ float approxU_dev(float x, float y) {
  float r2 = x*x + y*y;
  float dr2 = Rmaxdev*Rmaxdev / ((double) NRS);
  float idr2 = 1.0/dr2;
  int k = ((int) r2 *idr2);
  float u, u0, u6, u12;

  float t = atan2f(y, x);

  float C1, C2;
  if(k == 0){
    C2 = r2 * idr2;
    u0 =  C2 * U0dev[0];
    u6 =  C2 * U6dev[0];
    u12 = C2 * U12dev[0];
  }else{
    if(k >= NRS) k = NRS - 1;
    double r21 = dr2 * ((double) k);
    double r22 = dr2 * ((double) (k+1));
    C1 = (r22 - r2) * idr2;
    C2 = (r2 - r21) * idr2;
    u0 =  C1 * U0dev[k-1]  + C2 * U0dev[k];
    u6 =  C1 * U6dev[k-1]  + C2 * U6dev[k];
    u12 = C1 * U12dev[k-1] + C2 * U12dev[k];
  }

  u = u0 + u6 * cos(6.0*t) + u12 *cos(12.0*t);
  
  return u;
}


__device__ void accuObservs_dev(int N, int NT, float V0,
  float *X, float *Y, struct Observs obs)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float dx, dy;
  double V;
  float dx0, dy0;

  dx0 = X[tid] - XXdev[0];
  dy0 = Y[tid] - YYdev[0];

  float PP0xx=0, PP0yy=0, PP0xy=0;
  double Ex0, Ey0;
  float Xi, Yi;

  Xi = X[tid];
  Yi = Y[tid];

  V = 0.0;

  Ex0 = 0;
  Ey0 = 0;

  float Xs, Ys;
  Xs = X[tid+NT];
  Ys = Y[tid+NT];
  for(int j = 1; j<N; ++j){
    float Xj, Yj, invrij;
    
    int idj;
    Xj = Xs;
    Yj = Ys;
    if(j < N-1){
      idj = tid + (j+1)*NT;
      Xs = X[idj];
      Ys = Y[idj];
    }
    idj = tid + j*NT;

    dx = Xi - Xj;
    dy = Yi - Yj;

    invrij = rsqrtf(dx*dx+dy*dy);

    V += invrij;
    
    /* Setup PP */
    float ax, ay, temp;
    float invrij2, invrij3;
    invrij2 = invrij * invrij;
    invrij3 = invrij * invrij2;
    
    ax = invrij3 * dx;
    Ex0 += ax - PP0dev[j  ]*dx - PP0dev[j+2*N]*dy;
    ay = invrij3 * dy;
    Ey0 += ay - PP0dev[j+N]*dy - PP0dev[j+2*N]*dx;

    temp = 3.0f * invrij2;
    ax *= temp;
    ay *= temp;

    temp = invrij3 - ax * dx - PP0dev[j];
    obs.PP[0][idj] += temp;
    PP0xx -= temp;

    temp = invrij3 - ay * dy - PP0dev[j+N];
    obs.PP[1][idj] += temp;
    PP0yy -= temp;

    temp = - ax * dy - PP0dev[j+2*N];
    obs.PP[2][idj] += temp;
    PP0xy -= temp;
  }
  obs.PP[0][tid] += PP0xx;
  obs.PP[1][tid] += PP0yy;
  obs.PP[2][tid] += PP0xy;

  obs.AA[0][tid] += dx0*dx0;
  obs.AA[1][tid] += dy0*dy0;
  obs.AA[2][tid] += dx0*dy0;
  obs.AA[3][tid] += dy0*dx0;

  obs.PR[0][tid] +=  Ex0 * dx0;
  obs.PR[1][tid] +=  Ey0 * dy0;
  obs.PR[2][tid] +=  Ex0 * dy0;
  obs.PR[3][tid] +=  Ey0 * dx0;

  /* Determine V */

  V += approxU_dev(Xi, Yi);

  obs.Eg[tid] += V;
  
  for(int i = 1; i<N; ++i){
    int idi;
    float Xi, Yi;

    idi = tid + i*NT;
    Xi = X[idi];
    Yi = Y[idi];

    float Xs, Ys;
    idi = tid + (i+1)*NT;
    Xs = X[idi];
    Ys = Y[idi];
    for(int j = i+1; j<N; ++j){
      float Xj, Yj;
      float invrij;

      Xj = Xs;
      Yj = Ys;
      if(j < N-1){
       int idj = tid + (j+1)*NT;
       Xs = X[idj];
       Ys = Y[idj];
     }
     
     dx = Xi - Xj;
     dy = Yi - Yj;
     
     invrij = rsqrtf(dx*dx+dy*dy);

     V += invrij;
   }

   dx = Xi - XXdev[i];
   dy = Yi - YYdev[i];

    /* Correction to the finite simulation cell */
   V += approxU_dev(Xi, Yi);
 }

 V -= V0;
 
  /* Setup A, RR, PR */
 obs.RR[0][tid] += dx0*dx0*V;
 obs.RR[1][tid] += dy0*dy0*V;
 obs.RR[2][tid] += dx0*dy0*V;
 obs.RR[3][tid] += dy0*dx0*V;

 for(int i  = 1; i<N; ++i){
  int idi = tid + i*NT;
  dx = X[idi] - XXdev[i];
  dy = Y[idi] - YYdev[i];

  float temp;
  temp = dx * dx0;
  obs.AA[0][idi] += temp;
  obs.RR[0][idi] += temp*V;

  temp = dy * dy0;
  obs.AA[1][idi] += temp;
  obs.RR[1][idi] += temp*V;

  temp = dx * dy0;
  obs.AA[2][idi] += temp;
  obs.RR[2][idi] += temp*V;

  temp = dy * dx0;
  obs.AA[3][idi] += temp;
  obs.RR[3][idi] += temp*V;
  
  obs.PR[0][idi] +=  Ex0 * dx;
  obs.PR[1][idi] +=  Ey0 * dy;
  obs.PR[2][idi] +=  Ex0 * dy;
  obs.PR[3][idi] +=  Ey0 * dx;
}

obs.V[tid] += V;
}



__global__ void metropolis_dev(int N, int NT, int Nc, float *X, float *Y,
  struct Observs obs, float V0,
  tinymt32_status_t * tinymt, int M, float d,
  int STRIDE, int RUNS, long *NACCEP)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int NACCEPs = 0, i, ic, ep, ep1;
  float rr1, rr2;
  float X0, Y0, X1, Y1, Xi, Yi, XXi, YYi, r1, r2, dx, dy, dx0, dy0, p;
  unsigned int idx;
  tinymt32_status_t* mt;

  mt = tinymt + tid;
  
  for(int irun = 0; irun < RUNS; ++irun){
    for(int j = 0; j < STRIDE*N; ++j){
      rr1 = tinymt32_single(mt);
      ic = ((int) (N * rr1));
      
      idx = tid + ic*NT;
      X0 = X[idx];
      Y0 = Y[idx];

      rr1 = 1.0f - tinymt32_single(mt);
      rr1 = sqrtf(-2.0f*logf(rr1)) * d;

      rr2 = tinymt32_single(mt);
      rr2 *= TWOPI;

      sincosf(rr2, &dx, &dy);
      X1 = X0 + rr1 * dy;
      Y1 = Y0 + rr1 * dx;

      r1 = 1.0f;
      r2 = 1.0f;
      ep = 0.0f;
      float Xs = X[tid], Ys = Y[tid];
      for(i = 0; i < N; ++i){
       XXi = XXdev[i];
       YYi = YYdev[i];
       Xi = Xs;
       Yi = Ys;
       if(i < N-1){
         idx = tid + (i+1)*NT;
         Xs = X[idx];
         Ys = Y[idx];
       }
       if(i != ic){
         r1 *= (X1- Xi)*(X1- Xi) + (Y1- Yi)*(Y1- Yi);
         r1 *= (X0-XXi)*(X0-XXi) + (Y0-YYi)*(Y0-YYi);
         r1 = frexp(r1, &ep1);
         ep += ep1;
         r2 *= (X1-XXi)*(X1-XXi) + (Y1-YYi)*(Y1-YYi);
         r2 *= (X0- Xi)*(X0- Xi) + (Y0- Yi)*(Y0- Yi);
         r2 = frexp(r2, &ep1);
         ep -= ep1;
       }
     }

     XXi = XXdev[ic];
     YYi = YYdev[ic];

     dx = X1 - XXi;
     dy = Y1 - YYi;
     r1 *= normtheta1p_dev(dx *c1, dy *c1);
     
     dx0 = X0 - XXi;
     dy0 = Y0 - YYi;
     r2 *= normtheta1p_dev(dx0*c1, dy0*c1);

     p = r1/r2;
     p = scalbnf(p, ep);
     p = powf(p, M);

     p *= expf(c2x*(dx*dx - dx0*dx0) + c2y*(dy*dy - dy0*dy0));

     rr1 = tinymt32_single(mt);
     
     if(rr1 <= p){
       idx = tid + ic*NT;
       X[idx] = X1;
       Y[idx] = Y1;
       NACCEPs++;
     }
   }

   accuObservs_dev(Nc, NT, V0, X, Y, obs);
 }

 NACCEP[tid] += NACCEPs;
}


void HandleError( cudaError_t err,
  const char *file,
  int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
     file, line );
    exit( EXIT_FAILURE );
  }
}



int main(int argc, char **argv)
{
  double nu, a, d, Rc, Rc0;
  int M, NSTATS, NBT, NB, NT, RUNS, STRIDE;
  int NN, Nc, No, N1, N2;
  long STEPS0, STEPS, STEPSH, STEPSD, STEPU;
  double X0[NMAX], Y0[NMAX], XX[NMAX], YY[NMAX];
  int Rotate[NMAX], Reflect[NMAX], Equiv[NMAX];
  int device;
  const char *tinymtparamsfile;

  /* Parameters */
  // device = 1;
  tinymtparamsfile = "../tinymt32dc.txt";

  // nu = 0.120;
  // M  = 4;
  if(argc < 4){
    printf("Too few parameters:\n");
    printf("cfwccuda <device> <M> <nu>\n");
    exit(1);
  }

  device = (int) strtol(argv[1], NULL, 10);
  M = (int) strtol(argv[2], NULL, 10);
  nu = strtod(argv[3], NULL);

  printf("Device = %d, M = %d, nu = %f\n", device, M, nu);
  
  STEPS0 = 1000000;   // Burn-in number of samples
  //STEPS0 = 10000;
  STEPS  = 220;       // Number of steps 

  // 23 : 8  -- 227 : 80 
  STEPSD = 25;     //   Ajust these two parameters to make CPU and GPU load balanced.
  STEPSH =  5;     //   They also control the depth of MC sub-branch.

  STEPU = 100000;
  //STEPU = 1000;

  NSTATS = 10;    

  // 32x64  for GT650M
  // 70x160 for Tesla C2075
  // 15x960 for Tesla K40
  NBT = 960;  // These two parameters should be adjusted to maximize performance
  NB  = 26;   // of GPU

  RUNS   = 5; // Runs of MC simulation for each of kernel call -- should not be too large
  STRIDE = 2; // Steps betwen observables are evaluated

  Rc  = 11.0; // Number of hexagonal rings of the simulation cell   
  Rc0 =  9.0; // Number of hexagonal rings for evaluating observables 

  /* End parameters */
  
  Rc *= HALFSQRT3;
  Rc0*= HALFSQRT3;
  
  a = sqrt(4.0 * PI / sqrt(3.0)/nu);
  d = 1.0;
  NT = NB * NBT;

  /* Setup lattice */
  createlattice_(&Rc, &Rc0, &a, &Nc, &No, &N1, &N2, XX, YY, Rotate, Reflect, Equiv);

  NN = Nc + No;
  
  float V0 = 0;
  for(int i = 0; i<Nc; ++i){
    for(int j = i+1; j<Nc; ++j)
      V0 += 1.0/sqrt((XX[i] - XX[j])*(XX[i] - XX[j]) + (YY[i] - YY[j])*(YY[i] - YY[j]));
  }

  /* Determine ground state energy of a usual WC */
  double Eg0 = 0;

  int max =  ((int) ceil(6.0*a /TWOPI)) + 1;
  for(int k = -max; k<=max; ++k){
    for(int l = -max; l<=max; ++l){
      double KX = (TWOPI/a) * ((double) k);
      double KY = (TWOPI/a) * INVSQRT3 * ((double) (2*l - k));
      double K2 = KX*KX + KY*KY;
      if((k != 0 || l !=0) && K2 <= 25.0){
       double iK = 1.0/sqrt(K2);
       double K = 1.0 /iK;
       Eg0 += TWOPI * iK * exp(-K*K) ;
     }
   }
 }
 Eg0 /= HALFSQRT3 * (a*a); 
 Eg0 -=  EwaldU(0.0, 0.0, Nc, Rc0, a, 0.5*a, XX, YY);
  Eg0 -= sqrt(PI)*0.5;   // Self-interaction deduction
  Eg0 *= 0.5; //Note that there will be a correction of zero-point of the energy.

  /*
  timeval begin2, end2;
  double R = (Rc0 + 1) * a;
  double z0 = EwaldU(2.0*INVSQRT3 * R, 0.0, Nc, Rc0, a, 0.5*a, XX, YY);
  int NR = 100000;
  gettimeofday(&begin2, NULL);
  for(int i = 0; i < NR; ++i){
    double t = genrand_real2_();
    R =  genrand_real2_() *(Rc0 + 1.0) * a;
    double x = (1.0 - 0.5*t) * 2.0*INVSQRT3 * R;
    double y = t * R;
    double z = EwaldU(x, y, Nc, Rc0, a, 0.5*a, XX, YY);
    //    printf("%f %f  %f  %e\n", t, x, y, z - z0);
  }
  gettimeofday(&end2, NULL);

  printf("Duration = %f, %e calls/s\n", Duration(begin2, end2),
	 NR/Duration(begin2, end2)/((double) Nc));

  exit(0);
  */
  
  /* Allocate host memmory */
  float *X, *Y, *obsbuf;
  struct Observs obs;
  long *NACCEPN;
  X   = (float *) malloc(2*NT*NN*sizeof(float));
  Y   = X + NT*NN;
  obsbuf = (float *) malloc(NT*(15*Nc+2)*sizeof(float));
  setObservs(Nc, NT, &obs, obsbuf);
  NACCEPN = (long *) malloc(NT*sizeof(long));

  /* Setup PP0 */
  double *PP0;
  PP0 = (double *) malloc(3*Nc*sizeof(double));

  for(int i = 1; i < Nc; ++i){
  	double R = sqrt(XX[i]*XX[i] + YY[i]*YY[i]);
  	PP0[i]      = 1.0/pow(R, 3) - 3.0 * XX[i]*XX[i]/pow(R, 5);
  	PP0[i+Nc]   = 1.0/pow(R, 3) - 3.0 * YY[i]*YY[i]/pow(R, 5);
  	PP0[i+2*Nc] =               - 3.0 * XX[i]*YY[i]/pow(R, 5);
  }

  /* Burn-in, determine optimal d */
  unsigned int s = 13881;
  init_genrand_(&s);
  for(int i = 0; i<NN; ++i){
    X0[i] = XX[i] + d * (genrand_real2_() - 0.5);
    Y0[i] = YY[i] + d * (genrand_real2_() - 0.5);
  }

  long NACCEP;
  timeval end, begin;
  float duration, paccep;
  gettimeofday(&begin, NULL);
  for(int i = 0; i<NN; ++i){
    NACCEP = 0;
    metropolis_host_(&NN, X0, Y0, XX, YY, &M, &d, &a, &STEPS0, &NACCEP);
    d *= exp(0.1 * (((double) NACCEP) / ((double) STEPS0) - 0.25));
  }
  gettimeofday(&end, NULL);
  duration = Duration(begin, end);
  paccep = ((double) NACCEP) / ((double) STEPS0);
  printf("d = %f\tAcceptance Rate = %f\t", d, paccep);
  printf("Duration = %f\tSpeed = %f\n", duration, ((float) STEPS0)/duration);


  /* Setup interpolation function for U */
  double *U0, *U6, *U12, U00, Rmax;
  timeval begin1, end1;
  U0  = (double *) malloc(NRS * sizeof(double));
  U6  = (double *) malloc(NRS * sizeof(double));
  U12 = (double *) malloc(NRS * sizeof(double));

  Rmax = Rc0 * a;
  SampleEwaldU0(NRS, Nc, Rc0, a, XX, YY, Rmax, U0, U6, U12);

  FILE *fp1;

  fp1 = fopen("U06_0.dat", "w");
  for(int i = 0; i<NRS; ++i){
    double r = sqrt(Rmax * Rmax / ((double) NRS) * ((double) (i+1)));
    fprintf(fp1, "%d %f %e %e %e\n", i, r, U0[i], U6[i], U12[i]);
  }
  fclose(fp1);
  
  gettimeofday(&begin1, NULL);
  SampleEwaldU(NRS, STEPU, Nc, Rc0, a, d, M, X0, Y0, XX, YY, Rmax,
    &U00, U0, U6, U12);
  gettimeofday(&end1, NULL);
  
  double alpha = U0[0] * ((double) NRS) / (Rmax * Rmax);
  Eg0 -= U00 * 0.5;    // Adjust energy zero-point
  
  fp1 = fopen("U06_1.dat", "w");
  for(int i = 0; i<NRS; ++i){
    double r = sqrt(Rmax * Rmax / ((double) NRS) * ((double) (i+1)));
    fprintf(fp1, "%d %f %e %e %e\n", i, r, U0[i], U6[i], U12[i]);
  }
  fclose(fp1);

  printf("U setup completed. Duration: %f\n", Duration(begin1, end1));
  
  /*
  double u0 = EwaldU(0.0, 0.0, Nc, Rc0, a, 0.5*a, XX, YY);
  for(int i = 0; i<10000; ++i){
    double r = genrand_real2_() * (Rc0+0.5) * a;
    // double r = Rmax;
    double t = genrand_real2_() * TWOPI;
    double x = r * cos(t), y = r * sin(t);
    double u = approxU(x, y, NRS, Rmax, U0, U6, U12);
    //double u  = EwaldU(x, y, Nc, Rc0, a, 0.1*a, XX, YY) - u0;
    double u1 = EwaldU(x, y, Nc, Rc0, a, 0.5*a, XX, YY) - u0;
    printf("%f %f %e %e %e\n", r, t, u, u1, u-u1);
  }
  exit(1);
  */
  
  /* Setup device */
  timeval end0, endh, begin0;
  gettimeofday(&begin0, NULL);

  HANDLE_ERROR( cudaSetDevice(device) );
  
  /* Initialize Device RNG */
  unsigned int *seed;
  seed = (unsigned int *) malloc(NT*sizeof(unsigned int));
  for(int i = 0; i<NT; ++i){
    seed[i] = genrand_int32_();
  }

  tinymt32_status_t *mt;
  init_genrand_dev(tinymtparamsfile, seed, NB, NBT, &mt);
  free(seed);
  
  /* Allocate device memory */
  float *Xdev, *Ydev, *obsdevbuf;
  struct Observs obsdev;
  long  *NACCEPNdev;

  HANDLE_ERROR( cudaMalloc((void **) &Xdev,   2*NT*NN*sizeof(float)) );
  Ydev = Xdev + NT*NN;
  HANDLE_ERROR( cudaMalloc((void **) &obsdevbuf, NT*(15*Nc+2)*sizeof(float)) );
  setObservs(Nc, NT, &obsdev, obsdevbuf);
  HANDLE_ERROR( cudaMalloc((void **) &NACCEPNdev, NT*sizeof(long)) );

  /* Setup device constants */
  float *temp;
  temp = (float *) malloc(NN*sizeof(float));
  for(int i = 0; i<NN; ++i) temp[i] = (float) XX[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(XXdev,  temp,  NN*sizeof(float)) );
  for(int i = 0; i<NN; ++i) temp[i] = (float) YY[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(YYdev,  temp,  NN*sizeof(float)) );
  free(temp);

  temp = (float *) malloc(NRS*sizeof(float));
  for(int i = 0; i<NRS; ++i) temp[i] = (float) U0[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(U0dev, temp,  NRS*sizeof(float)) );
  for(int i = 0; i<NRS; ++i) temp[i] = (float) U6[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(U6dev, temp,  NRS*sizeof(float)) );
  for(int i = 0; i<NRS; ++i) temp[i] = (float) U12[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(U12dev, temp,  NRS*sizeof(float)) );
  free(temp);

  temp = (float *) malloc(3*Nc*sizeof(float));
  for(int i = 0; i<3*Nc; ++i) temp[i] = (float) PP0[i];
    HANDLE_ERROR( cudaMemcpyToSymbol(PP0dev, temp,  3*Nc*sizeof(float)) );
  free(temp);  

  assert(M >= 0);

  float c1h, c2xh, c2yh;
  c1h = PI/a;
  c2xh = -0.5 + (2.0*PI/sqrt(3.0)/(a*a)) * ((float) M) ;
  c2yh = -0.5 - (2.0*PI/sqrt(3.0)/(a*a)) * ((float) M);
  HANDLE_ERROR( cudaMemcpyToSymbol(c1,  &c1h,  sizeof(float)) );
  HANDLE_ERROR( cudaMemcpyToSymbol(c2x, &c2xh, sizeof(float)) );
  HANDLE_ERROR( cudaMemcpyToSymbol(c2y, &c2yh, sizeof(float)) );
  float Rmaxf = Rmax;
  HANDLE_ERROR( cudaMemcpyToSymbol(Rmaxdev, &Rmaxf, sizeof(float)) );

  /* Start the simulation */

  long TOTALSTEPS=0;
  NACCEP = 0;
  HANDLE_ERROR( cudaMemset(NACCEPNdev, 0, NT*sizeof(long)) );

  double *buf1;
  struct ObservsDouble obs0, aobs, dobs;
  buf1 = (double *) malloc(3*(15*Nc+2)*sizeof(double));
  setObservsDouble(Nc, &obs0,  buf1);
  setObservsDouble(Nc, &aobs, buf1 + 15*Nc+2);
  setObservsDouble(Nc, &dobs, buf1 + 30*Nc+4);

  /* Initialize X and Y */
  int idx;
  long STEPS3 = NN * STRIDE;
  memset(obs0.AA[0], 0, (15*Nc+2)*sizeof(double));
  for(int j = 0; j < NT; ++j){
    for(int i = 0; i < STEPSH; ++i){
      metropolis_host_(&NN, X0, Y0, XX, YY, &M, &d, &a, &STEPS3, &NACCEP);
      accuObservs_host(Nc, ((double) V0), X0, Y0, XX, YY,
       NRS, Rmax, U0, U6, U12, PP0, obs0);
    }
    for(int i = 0; i<NN; ++i){
      idx = j + i*NT;
      X[idx] = X0[i];
      Y[idx] = Y0[i];
    }
  }
  // We need to guess a value for the averaged V, to improve the accuracy.
  V0 +=  (obs0.V[0]/((double) NT*STEPSH));

  printf("NN = %d, Nc = %d, Eg0 = %f, V0 = %f, alpha = %e\n", NN, Nc, Eg0, V0, alpha);
  
  memset(aobs.AA[0], 0, (30*Nc+4)*sizeof(double));

  for(int nn = 0; nn<NSTATS; ++nn){
    memset(obs0.AA[0], 0, (15*Nc+2)*sizeof(double));

    long TOTALSTEPS1 = 0;
    for(int ns = 0; ns < STEPS; ++ns){
      gettimeofday(&begin, NULL);

      /* Device execution */
      HANDLE_ERROR( cudaMemcpy(Xdev,   X, 2*NT*NN*sizeof(float), cudaMemcpyHostToDevice) );
      HANDLE_ERROR( cudaMemset(obsdev.AA[0], 0, NT*(15*Nc+2)*sizeof(float)) );

      for(int i = 0; i<STEPSD; ++i)
       metropolis_dev<<<NB, NBT>>>(NN, NT, Nc, Xdev, Ydev, obsdev, ((float) V0),
        mt, M, ((float) d), STRIDE, RUNS, NACCEPNdev);

      /* Host execution */
     for(int j = 0; j < NT; ++j){
       for(int i = 0; i < STEPSH; ++i){
        metropolis_host_(&NN, X0, Y0, XX, YY, &M, &d, &a, &STEPS3, &NACCEP);
	  		//accuObservs_host(Nc, ((double) V0), X0, Y0, XX, YY,
	  		//		   NRS, Rmax, U0, U6, U12, PP0, obs0);
      }
      for(int i = 0; i<NN; ++i){
       idx = j + i*NT;
       X[idx] = X0[i];
       Y[idx] = Y0[i];
     }
   }
   gettimeofday(&endh, NULL);

      /* Synchronize host and device */
   HANDLE_ERROR( cudaDeviceSynchronize() );

      /* Combine host and device */ 
   HANDLE_ERROR( cudaMemcpy(obs.AA[0], obsdev.AA[0], NT*(15*Nc+2)*sizeof(float),
    cudaMemcpyDeviceToHost) );

   collectObservs(Nc, NT, obs, obs0);

   gettimeofday(&end, NULL);
   
   TOTALSTEPS1 += (STEPSD * RUNS + 0*STEPSH)*NT;
   duration = Duration(begin, end);
   printf("STEPS: %11ld Duration = %f\tDSpeed = %f\tTD = %f\tHSpeed = %f\tAverage = %f,%e\n",
    TOTALSTEPS + TOTALSTEPS1, duration,
    ((double) (STEPSH + STEPSD*RUNS)*NT*STRIDE)/duration, Duration(endh, end),
    ((double) STEPSH*NT*STRIDE)/Duration(begin, endh),
    (obs0.AA[0][0] + obs0.AA[1][0])/((double) TOTALSTEPS1),
    (obs0.Eg[0]/((double) TOTALSTEPS1)) * 0.5 - Eg0);
 }
 TOTALSTEPS += TOTALSTEPS1 + STEPSH*NT*STEPS;
 averageObservs(Nc, TOTALSTEPS1, obs0, aobs, dobs);
}

normalizeObservs(Nc, NSTATS, alpha, V0, aobs, dobs);

HANDLE_ERROR( cudaMemcpy(NACCEPN, NACCEPNdev, NT*sizeof(long), cudaMemcpyDeviceToHost) );

for(int i = 0; i < NT; ++i) NACCEP += NACCEPN[i];

  gettimeofday(&end0, NULL);

outputObservs("AA", Nc, M, nu, XX,  YY,
  aobs.AA[0], aobs.AA[1], aobs.AA[2], aobs.AA[3],
  dobs.AA[0], dobs.AA[1], dobs.AA[2], dobs.AA[3]);
double *PP3;
PP3 = (double *) malloc(Nc*sizeof(double));
memcpy(PP3, aobs.PP[2], Nc*sizeof(double));
outputObservs("PP", Nc, M, nu, XX,  YY,
  aobs.PP[0], aobs.PP[1], aobs.PP[2], PP3,
  dobs.PP[0], dobs.PP[1], dobs.PP[2], dobs.PP[2]);
free(PP3);
outputObservs("PR", Nc, M, nu, XX,  YY,
  aobs.PR[0], aobs.PR[1], aobs.PR[2], aobs.PR[3],
  dobs.PR[0], dobs.PR[1], dobs.PR[2], dobs.PR[3]);

outputObservs("RR", Nc, M, nu, XX,  YY,
  aobs.RR[0], aobs.RR[1], aobs.RR[2], aobs.RR[3],
  dobs.RR[0], dobs.RR[1], dobs.RR[2], dobs.RR[3]);

duration = Duration(begin0, end0);
printf("Acceptance Rate = %f\t<r^2> = %f\t  D_Eg/N = %f\n",
  ((float) NACCEP)/((float) TOTALSTEPS*NN*STRIDE),
  aobs.AA[0][0] + aobs.AA[1][0],
  aobs.Eg[0] - Eg0);

printf("Total Time = %f \t Speed = %f \n",  duration, ((float) TOTALSTEPS*STRIDE)/duration);

FILE *fp;
char f0[256];

sprintf(f0, "Parameters-%1d-%6.4f", M, nu);
fp = fopen(f0, "w");
fprintf(fp, "nu = %f, M = %d, D_Eg = %e (+/-) %e\n", nu, M, aobs.Eg[0] - Eg0, dobs.Eg[0]);
fprintf(fp, "Total number of samples: %ld\n", TOTALSTEPS);
fprintf(fp, "Rc = %f,  NN = %d,  Nc = %d\n", Rc, NN, Nc);
fprintf(fp, "alpha = %e\n", alpha);
fclose(fp);


free(X);
free(obsbuf);
free(NACCEPN);
free(buf1);

HANDLE_ERROR( cudaFree(Xdev) );
HANDLE_ERROR( cudaFree(obsdevbuf) );
HANDLE_ERROR( cudaFree(NACCEPNdev) );

}
