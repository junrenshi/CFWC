#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cfwc.h"


inline double rsqrt64(const double number)
{
  unsigned long i;
  double x2, y;
  x2 = number * 0.5;
  y = number;
  i = *(unsigned long *) &y;
  i = 0x5fe6eb50c7b537a9 - (i >> 1);
  y = *(double *) &i;
  y = y * (1.5 - (x2 * y * y));
  y = y * (1.5 - (x2 * y * y));
  y = y * (1.5 - (x2 * y * y));
  return y;
}

/* The function calculate potential insided a 2D hollow lattice */
double EwaldU(double x, double y, int N, double Rc, double a, double eta,
	      double *XX, double *YY){

  double U = 0.0;
  double dx, dy, r, ir, ieta;
  double eta2;
  
  ieta = 0.5/eta;  // alpha = 1/(2 eta) 
  eta2 = eta * eta;
  
  for(int i=0; i<N; ++i){
    dx = x - XX[i];
    dy = y - YY[i];
    double r2=dx*dx + dy*dy;
    if(r2  > 1e-16 * eta2 ){
      ir = rsqrt64(r2);
      if(r2 > 100 * eta2){
	U -= ir;
      }else{
	r = 1.0 / ir;
	U -= ir * erf(r * ieta);
      }
    }else{
      U -= 1.12837916710 * ieta;
    }
  }

  int m = (int) round((x - y * INVSQRT3)/a);
  int n = (int) round(2.0 * INVSQRT3 * y/a);
  double Rmax = 10 * eta;
  int max = ((int) ceil(1.2*Rmax/a))+1;
  double eps = 1e-6;
  
  for(int i = -max+m; i<=max+m; ++i){
    for(int j = -max+n; j<=max+n; ++j){
      double X = ((double) i + ((double) j) * 0.5);
      double Y = ((double) j) * HALFSQRT3;
      dx = x - X * a;
      dy = y - Y * a;

      if( dx*dx + dy*dy  < Rmax * Rmax &&
	  (fabs(HALFSQRT3*X+0.5*Y) > Rc+eps  ||
	   fabs(HALFSQRT3*X-0.5*Y) > Rc+eps  ||
	   fabs(Y)                 > Rc+eps ))
	{
	  ir = rsqrt64(dx*dx + dy*dy);
	  double r = 1.0/ir;
	  U += ir * erfc(r * ieta);
	}
    }
  } 

  double KX, KY;
  double iK, K;
  
  double Kmax = 10.0 * ieta, U1;
  max = ((int) ceil(1.2*a /TWOPI * Kmax)) + 1;
  double K0X = TWOPI/a, K0Y = K0X * INVSQRT3;

  double omega = TWOPI /(HALFSQRT3 * a * a);
  for(int k = -max; k<=max; ++k){
    for(int l = -max; l<=max; ++l){
      KX = K0X * ((double) k);
      KY = K0Y * ((double) (2*l - k));
      double K2 = KX*KX + KY*KY;
      if((k != 0 || l !=0) && K2 <= Kmax*Kmax){
	iK = rsqrt64(K2);
	K = 1.0 /iK;
	U += omega * iK * cos(KX*x + KY*y) *  erfc(K*eta);
      }
    }
  }

  U -= 4.0 * sqrt(PI) * eta / (a*a) * 2.0 * INVSQRT3;;

  return U;
}
  

void SampleEwaldU0(int NRS, int N, double Rc, double a, double *XX, double *YY, 
		  double Rmax, double *U0, double *U6, double *U12){
  double dr2 = Rmax*Rmax / ((double) NRS);
  int NH = ((int) (Rc * 2.0 * INVSQRT3)) * 6;
  int NTH = NH * 8;
  double dt = TWOPI / ((double) NTH);
  double u00;

  u00 = EwaldU(0.0, 0.0, N, Rc, a, 0.5*a, XX, YY);
  
  for(int j = 0; j < NRS; ++j){
    double r2 = dr2 * ((double) (j+1));
    double r = sqrt(r2);
    double u0 = 0.0;
    double u6 = 0.0;
    double u12 = 0.0;
    for(int i = 0; i < NTH; ++i){
      double t = dt * ((double) i);
      double x = r * cos(t);
      double y = r * sin(t);
      double u;

      u = EwaldU(x, y, N, Rc, a, 0.5*a, XX, YY) - u00;
      u0   += u;
      u6   += u * cos(6.0*t);
      u12  += u * cos(12.0*t);
    }

    U0[j]  = u0  / ((double) NTH);
    U6[j]  = u6  / ((double) NTH) * 2.0;
    U12[j] = u12 / ((double) NTH) * 2.0;
  }
}


void SampleEwaldU(int NRS, long STEPU, int N, double Rc, double a, double d, int M,
		  double *X0, double * Y0, double *XX, double *YY, 
		  double Rmax, double *U00, double *U0, double *U6, double *U12){
  double dr2 = Rmax*Rmax / ((double) NRS);
  int NTH = 5;
  double dt = PI / ((double) (6*(NTH-1)));
  double u00;

  /* Setup coordinate table */

  double *x, *y, *u;
  x = (double *) malloc(NRS*NTH*sizeof(double));
  y = (double *) malloc(NRS*NTH*sizeof(double));
  u = (double *) malloc(NRS*NTH*sizeof(double));
  
  for(int j = 0; j<NTH; ++j){
    double t = dt * ((double) j);
    double c=cos(t), s=sin(t);
    for(int i = 0; i<NRS; ++i){
      int id = i + j*NRS;
      double r2 = dr2 * ((double) (i+1));
      double r = sqrt(r2);
      x[id] = r * c;
      y[id] = r * s;
    }
  }

  int *INNER, NI;
  INNER = (int *)malloc(N*sizeof(int));
  NI = 0;
  for(int i = 0; i<N; ++i){
    if(XX[i]*XX[i] + YY[i]*YY[i] < (Rc*Rc)*(a*a)*0.1){
      INNER[NI] = i;
      NI++;
    }
  }

  long NACCEP, STRIDE = 4*N;
  double x0, y0;

  u00 = 0.0;
  memset(u, 0, NRS*NTH*sizeof(double));

  double c = cos(PI / 3.0), s = sin(PI / 3.0);
  for(long i = 0; i<STEPU; ++i){
    metropolis_host_(&N, X0, Y0, XX, YY, &M, &d, &a, &STRIDE, &NACCEP);

    for(int j = 0; j < NI; ++j){
      x0 = X0[INNER[j]] - XX[INNER[j]];
      y0 = Y0[INNER[j]] - YY[INNER[j]];

      for(int irfl = 0; irfl <= 1; ++irfl){
	for(int irot = 0; irot <6; ++irot){
	  u00 += approxU(x0, y0, NRS, Rmax, U0, U6, U12);
	  for(int k = 0; k<NRS*NTH; ++k){
	    u[k] += approxU(x[k]+x0, y[k]+y0, NRS, Rmax, U0, U6, U12);
	  }
	  double x1 = x0, y1 = y0;
	  x0 = c * x1 - s * y1;
	  y0 = s * x1 + c * y1;
	}
	y0 = - y0;
      } //irfl
    }//j
  }//i

  u00 /= ((double) (12*STEPU*NI));
  for(int i = 0; i<NRS*NTH; ++i){
    u[i] /= ((double) (12*STEPU*NI));
  }
  
  for(int i = 0; i < NRS; ++i){
    double u0 = 0.0;
    double u6 = 0.0;
    double u12 = 0.0;
    for(int j = 0; j < NTH; ++j){
      double t = dt * ((double) j);
      double uu;
      int id = i + j*NRS;
      uu = u[id];
      if( j == 0 || j == NTH-1) uu *= 0.5;
      
      u0   += uu;
      u6   += uu * cos(6.0*t);
      u12  += uu * cos(12.0*t);
    }

    *U00   = u00; 
    U0[i]  = u0  / ((double) (NTH-1)) - u00;
    U6[i]  = u6  / ((double) (NTH-1)) * 2.0;
    U12[i] = u12 / ((double) (NTH-1)) * 2.0;
  }

  free(x);
  free(y);
  free(u);
  free(INNER);
}




void accu_approxU(int NRS, double Rmax, double x1, double y1,
		  double *U0, double *U6, double *U12,
		  double *UU00, double *UU0, double *UU6, double *UU12){
  double dr2 = Rmax*Rmax / ((double) NRS);
  int NTH = 24;
  double dt = TWOPI / ((double) NTH);

  for(int j = 0; j < NRS; ++j){
    double r2 = dr2 * ((double) (j+1));
    double r = sqrt(r2);
    double u0 = 0.0;
    double u6 = 0.0;
    double u12 = 0.0;
    for(int i = 0; i < NTH; ++i){
      double t = dt * ((double) i);
      double x = r * cos(t);
      double y = r * sin(t);
      double u;

      u = approxU(x+x1, y+y1, NRS, Rmax, U0, U6, U12);
      u0   += u;
      u6   += u * cos(6.0*t);
      u12  += u * cos(12.0*t);
    }

    *UU00 += approxU(x1, y1, NRS, Rmax, U0, U6, U12);
    UU0[j]  += u0  / ((double) NTH);
    UU6[j]  += u6  / ((double) NTH) * 2.0;
    UU12[j] += u12 / ((double) NTH) * 2.0;
  }
}



void accuObservs_host(int N, double V0, double *X, double *Y, double *XX, double * YY,
		      int NRS, double Rmax, double *U0, double *U6, double *U12, double *PP0,
		      struct ObservsDouble obs)
{
  double dx, dy, dx0, dy0;
  double V;

  dx0 = X[0] - XX[0];
  dy0 = Y[0] - YY[0];

  double PP0xx=0, PP0yy=0, PP0xy=0;
  double Ex0, Ey0;
  double Xi, Yi;

  Xi = X[0];
  Yi = Y[0];

  V = 0.0;
  Ex0 = 0;
  Ey0 = 0;

  for(int j = 1; j<N; ++j){
    double Xj, Yj, ir;
    
    Xj = X[j];
    Yj = Y[j];

    dx = Xi - Xj;
    dy = Yi - Yj;
	
    ir  = rsqrt64(dx *dx +dy *dy );

    V += ir;

    /* Setup PP */
    double ax, ay, temp;
    double ir2, ir3;
    ir2 = ir * ir;
    ir3 = ir * ir2;
	  
    ax = ir3 * dx;
    Ex0 += ax - PP0[j  ]*dx - PP0[j+2*N]*dy;
    ay = ir3 * dy;
    Ey0 += ay - PP0[j+N]*dy - PP0[j+2*N]*dx;

    temp = 3.0 * ir2;
    ax *= temp;
    ay *= temp;

    temp = ir3  - ax * dx - PP0[j];
    obs.PP[0][j] += temp;
    PP0xx -= temp;

    temp = ir3  - ay * dy - PP0[j+N];
    obs.PP[1][j] += temp;
    PP0yy -= temp;

    temp = - ax * dy - PP0[j+2*N];
    obs.PP[2][j] += temp;
    PP0xy -= temp;
  }
  obs.PP[0][0] += PP0xx;
  obs.PP[1][0] += PP0yy;
  obs.PP[2][0] += PP0xy;

  obs.AA[0][0] += dx0*dx0;
  obs.AA[1][0] += dy0*dy0;
  obs.AA[2][0] += dx0*dy0;
  obs.AA[3][0] += dy0*dx0;

  obs.PR[0][0] +=  Ex0 * dx0;
  obs.PR[1][0] +=  Ey0 * dy0;
  obs.PR[2][0] +=  Ex0 * dy0;
  obs.PR[3][0] +=  Ey0 * dx0;

  V += approxU(Xi, Yi, NRS, Rmax, U0, U6, U12);

  obs.Eg[0] += V;
  
  /* Determine V */

  for(int i = 1; i<N; ++i){
    double Xi, Yi;

    Xi = X[i];
    Yi = Y[i];

    for(int j = i+1; j<N; ++j){
      dx = Xi - X[j];
      dy = Yi - Y[j];
	
      V += rsqrt64(dx*dx+dy*dy);
    }

    dx = Xi - XX[i];
    dy = Yi - YY[i];
    V += approxU(Xi, Yi, NRS, Rmax, U0, U6, U12);
  }

  V -= V0;
  
  /* Setup A, RR, PR */
  obs.RR[0][0] += dx0*dx0*V;
  obs.RR[1][0] += dy0*dy0*V;
  obs.RR[2][0] += dx0*dy0*V;
  obs.RR[3][0] += dy0*dx0*V;

  for(int i  = 1; i<N; ++i){
    Xi = X[i];
    Yi = Y[i];
    dx = Xi - XX[i];
    dy = Yi - YY[i];

    float temp;
    temp = dx * dx0;
    obs.AA[0][i] += temp;
    obs.RR[0][i] += temp*V;

    temp = dy * dy0;
    obs.AA[1][i] += temp;
    obs.RR[1][i] += temp*V;

    temp = dx * dy0;
    obs.AA[2][i] += temp;
    obs.RR[2][i] += temp*V;

    temp = dy * dx0;
    obs.AA[3][i] += temp;
    obs.RR[3][i] += temp*V;
    
    obs.PR[0][i] +=  Ex0 * dx;
    obs.PR[1][i] +=  Ey0 * dy;
    obs.PR[2][i] +=  Ex0 * dy;
    obs.PR[3][i] +=  Ey0 * dx;
  }
  
  obs.V[0] += V;
}



float Duration(timeval begin, timeval end){
  return ((float) (end.tv_sec - begin.tv_sec)) + ((float) (end.tv_usec - begin.tv_usec))/1e6;
}

void setObservs(int N, int NT, struct Observs *obs, float *buf){
  float *p;
  
  p = buf;
  
  for(int a = 0; a<4; ++a){
    obs->AA[a] = p;
    p += NT*N;
  }
  for(int a = 0; a<3; ++a){
    obs->PP[a] = p;
    p += NT*N;
  }
  for(int a = 0; a<4; ++a){
    obs->PR[a] = p;
    p += NT*N;
  }
  for(int a = 0; a<4; ++a){
    obs->RR[a] = p;
    p += NT*N;
  }

  obs->V = p;
  p += NT;

  obs->Eg = p;
}

void setObservsDouble(int N, struct ObservsDouble *obs, double *buf){
  double *p;
  p = buf;
  for(int a = 0; a<4; ++a){
    obs->AA[a] = p;
    p += N;
  }
  for(int a = 0; a<3; ++a){
    obs->PP[a] = p;
    p += N;
  }
  for(int a = 0; a<4; ++a){
    obs->PR[a] = p;
    p += N;
  }
  for(int a = 0; a<4; ++a){
    obs->RR[a] = p;
    p += N;
  }

  obs->V = p;
  p++;

  obs->Eg = p;
}


void collectObservs(int N, int NT, struct Observs obs, struct ObservsDouble obs0){
  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i)
      for(int j = 0; j<NT; ++j)
	obs0.AA[a][i] += (double) obs.AA[a][j + i*NT];
      
  for(int a = 0; a<3; ++a)
    for(int i = 0; i< N; ++i)
      for(int j = 0; j<NT; ++j)
	obs0.PP[a][i] += (double) obs.PP[a][j + i*NT];

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i)
      for(int j = 0; j<NT; ++j)
	obs0.PR[a][i] += (double) obs.PR[a][j + i*NT];

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i)
      for(int j = 0; j<NT; ++j)
	obs0.RR[a][i] += (double) obs.RR[a][j + i*NT];

  for(int j = 0; j<NT; ++j)
    obs0.V[0] += (double) obs.V[j];

  for(int j = 0; j<NT; ++j)
    obs0.Eg[0] += (double) obs.Eg[j];
}


void averageObservs(int N, long STEPS,
		    struct ObservsDouble obs0,  struct ObservsDouble aobs,
		    struct ObservsDouble dobs){

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      aobs.AA[a][i] += obs0.AA[a][i]/((double) STEPS);
      dobs.AA[a][i] += pow(obs0.AA[a][i]/((double) STEPS), 2);
    }

  for(int a = 0; a<3; ++a)
    for(int i = 0; i< N; ++i){
      aobs.PP[a][i] += obs0.PP[a][i]/((double) STEPS);
      dobs.PP[a][i] += pow(obs0.PP[a][i]/((double) STEPS), 2);
    }
  
  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      aobs.PR[a][i] += obs0.PR[a][i]/((double) STEPS);
      dobs.PR[a][i] += pow(obs0.PR[a][i]/((double) STEPS), 2);
    }

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      aobs.RR[a][i] += obs0.RR[a][i]/((double) STEPS);
      dobs.RR[a][i] += pow(obs0.RR[a][i]/((double) STEPS), 2);
    }
  
  aobs.V[0] += obs0.V[0]/((double) STEPS);
  dobs.V[0] += pow(obs0.V[0]/((double) STEPS), 2);

  aobs.Eg[0] += obs0.Eg[0]/((double) STEPS);
  dobs.Eg[0] += pow(obs0.Eg[0]/((double) STEPS), 2);
}


void normalizeObservs(int N, int NSTATS, double alpha, double V0,
		      struct ObservsDouble aobs, struct ObservsDouble dobs){
  double t;
  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      t = aobs.AA[a][i] / ((double) NSTATS);
      aobs.AA[a][i] = t;
      dobs.AA[a][i] = sqrt((dobs.AA[a][i]/((double) NSTATS) - t*t)/((double ) (NSTATS-1)));
    }

  for(int a = 0; a<3; ++a)
    for(int i = 0; i< N; ++i){
      t = aobs.PP[a][i] / ((double) NSTATS);
      aobs.PP[a][i] = t;
      dobs.PP[a][i] = sqrt((dobs.PP[a][i]/((double) NSTATS) - t*t)/((double ) (NSTATS-1)));
    }

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      t = aobs.PR[a][i] / ((double) NSTATS);
      // This is correction to the finite simulation cell
      aobs.PR[a][i] = t;
      dobs.PR[a][i] = sqrt((dobs.PR[a][i]/((double) NSTATS) - t*t)/((double ) (NSTATS-1)));
    }

  for(int a = 0; a<4; ++a)
    for(int i = 0; i< N; ++i){
      t = aobs.RR[a][i] / ((double) NSTATS);
      aobs.RR[a][i] = t;
      dobs.RR[a][i] = sqrt((dobs.RR[a][i]/((double) NSTATS) - t*t)/((double ) (NSTATS-1)));
    }

  t = aobs.V[0] / ((double) NSTATS);
  aobs.V[0] = t;
  dobs.V[0] = sqrt((dobs.V[0]/((double) NSTATS) - t*t)/((double ) (NSTATS-1)));
    
  t = aobs.Eg[0] / ((double) NSTATS);
  aobs.Eg[0] = t  * 0.5;
  dobs.Eg[0] = sqrt((dobs.Eg[0]/((double) NSTATS) - t*t)/((double ) (NSTATS-1))) * 0.5;

  for(int a = 0; a<4; ++a)
    for(int i = 0; i<N; ++i){
      aobs.RR[a][i] -= aobs.V[0] * aobs.AA[a][i];
      dobs.RR[a][i] = sqrt(dobs.RR[a][i]*dobs.RR[a][i]
			   + (aobs.V[0]*aobs.V[0])* (dobs.AA[a][i]*dobs.AA[a][i])
			   + (dobs.V[0] * dobs.V[0]) * (aobs.AA[a][i] * aobs.AA[a][i]));
    }  
}


void outputObservs(const char *obsname, int N, int M, double nu,
		   double *XX, double *YY,
		   double * AAXX, double * AAYY, double * AAXY, double * AAYX,
		   double * DAXX, double * DAYY, double * DAXY, double * DAYX)
{
  FILE *fp;
  char f0[256], f1[256];

  sprintf(f0, "%s-%1d-%6.4f", obsname, M, nu);
  sprintf(f1, "%s-1.dat", f0);
  fp = fopen(f1, "w");
  for(int i = 0; i<N; ++i)
    fprintf(fp, "%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\t%14.6e\n",
	    XX[i], YY[i],
	    AAXX[i], AAYY[i], AAXY[i], AAYX[i],
	    DAXX[i], DAYY[i], DAXY[i], DAYX[i]);
  fclose(fp);
}

