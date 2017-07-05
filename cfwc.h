#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define PI     3.1415926535898f
#define TWOPI  6.2831853071796f
#define HALFPI 1.5707963267949f
#define HALFSQRT3 0.86602540378444
#define INVSQRT3  0.57735026918963


#define NMAX 512


/* Total length: NT * (15*N + 2) */
struct Observs{
  float *AA[4];  /* NT * N * 4 */ //  <xi  x0>
  float *PP[3];  /* NT * N * 3 */ //  < D_i0 >
  float *PR[4];  /* NT * N * 4 */ //  <xi  E0>
  float *RR[4];  /* NT * N * 4 */ // <xi V xi0 >
  float *V;      /* NT */         // Total energy 
  float *Eg;                      // Energy of the center electron 
}; 

/* Total length: 15*N + 2 */
struct ObservsDouble{
  double *AA[4];  /* N * 4 */
  double *PP[3];  /* N * 3 */
  double *PR[4];  /* N * 4 */
  double *RR[4];  /* N * 4 */
  double *V;      /* 1 */
  double *Eg;     /* Ground state energy */
}; 

extern "C" {
  void metropolis_host_(int *N, double *X, double *Y, double *XX, double *YY, int *M,
		       double *d, double *a, long *STEPS, long *NACCEP);

  void createlattice_(double *RC, double *Rc0, double *a, int *Nc, int *No, int *N1, int *N2,
		      double *XX, double *YY, int *Rotate, int *Reflect, int *Equiv);

  void symmetrizeaa_(int *N, int *N1, int *N2, double *XX, double *YY,
		     int *Rotate, int *Reflect, int *Equiv,
		    double *AAXX, double *AAYY, double *AAXY, double *AAYX);

  void  accua_(int *N, int *N0, double *X, double *Y, double *XX, double *YY,
	      double *AXX, double *AYY, double *AXY, double *AYX);

  void accuaa_(int *N, long *STEPS, double *AXX, double *AYY, double *AXY, double *AYX, 
	      double *AAXX, double *AAYY, double *AAXY, double *AAYX,
	      double *DAXX, double *DAYY, double *DAXY, double *DAYX);

  void normaa_(int *N, int *NSTATS, double *AAXX, double *AAYY, double *AAXY, double *AAYX, 
	      double *DAXX, double *DAYY, double *DAXY, double *DAYX);

  void init_genrand_(unsigned int *s);

  unsigned int genrand_int32_();

  unsigned int genrand_int31_();

  double genrand_real1_();

  double genrand_real2_();

  double genrand_real3_();

  double genrand_res53_();

  double EwaldU(double x, double y, int N, double Rc, double a, double eta,
		double *XX, double *YY);
  
  void SampleEwaldU0(int NRS, int N, double Rc, double a, double *XX, double *YY, 
		     double Rmax, double *U0, double *U6, double *U12);
  
  void SampleEwaldU(int NRS, long STEPU, int N, double Rc, double a, double d, int M,
		    double *X0, double * Y0, double *XX, double *YY, 
		    double Rmax, double *U00, double *U0, double *U6, double *U12);
  
  double approxU(double x, double y, int NRS, double Rmax,
		 double *U0, double *U6, double *U12);

  void accuObservs_host(int N, double V0, double *X, double *Y, double *XX, double * YY,
			int NRS, double Rmax, double *U0, double *U6, double *U12, double *PP0,
			struct ObservsDouble obs);

  float Duration(timeval begin, timeval end);

  void setObservs(int N, int NT, struct Observs *obs, float *buf);

  void setObservsDouble(int N, struct ObservsDouble *obs, double *buf);

  void collectObservs(int N, int NT, struct Observs obs, struct ObservsDouble obs0);

  void averageObservs(int N, long STEPS,
		      struct ObservsDouble obs0,  struct ObservsDouble aobs,
		      struct ObservsDouble dobs);

  void normalizeObservs(int N, int NSTATS, double alpha, double V0,
			struct ObservsDouble aobs, struct ObservsDouble dobs);

  void outputObservs(const char *obsname, int N, int M, double nu,
			double *XX, double *YY,
			double * AAXX, double * AAYY, double * AAXY, double * AAYX,
			double * DAXX, double * DAYY, double * DAXY, double * DAYX);
}

inline double approxU(double x, double y, int NRS, double Rmax,
	       double *U0, double *U6, double *U12) {
  double r2 = x*x + y*y;
  double dr2 = Rmax*Rmax / ((double) NRS);
  double idr2 = 1.0/dr2;
  double u, u0, u6, u12;
  int k = ((int) (r2 *idr2));

  double t = atan2(y, x);

  double c1, c2;
  if(k == 0){
    c2 = r2 * idr2;
    u0 =  c2 * U0[0];
    u6 =  c2 * U6[0];
    u12 = c2 * U12[0];
  }else{
    if(k >= NRS) k = NRS - 1;
    double r21 = dr2 * ((double) k);
    double r22 = dr2 * ((double) (k+1));
    c1 = (r22 - r2) * idr2;
    c2 = (r2 - r21) * idr2;
    u0 =  c1 * U0[k-1] + c2 * U0[k];
    u6 =  c1 * U6[k-1] + c2 * U6[k];
    u12 = c1 * U12[k-1] + c2 * U12[k];
  }

  u = u0 + u6 * cos(6.0*t) + u12 *cos(12.0*t);
  
  return u;
}


