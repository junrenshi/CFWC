
cfwccuda: cfwccuda.c tinymt32_kernel.c cfwccuda_host.o ipo_out.o
	nvcc -O3  cfwccuda.cu cfwccuda_host.o ipo_out.o -o cfwccuda -L/opt/intel/compilers_and_libraries/mac/lib -limf -lintlc -lsvml

cfwccuda_gcc: cfwccuda.c tinymt32_kernel.c cfwccuda_host.o cfwc_metropolis.o mt19937ar.o
	nvcc -O3  cfwccuda.cu cfwccuda_host.o cfwc_metropolis.o mt19937ar.o -o cfwccuda_gcc

cfwccuda_AAonly: cfwccuda_AAonly.c tinymt32_kernel.c cfwccuda_host.o ipo_out.o
	nvcc -O3  cfwccuda_AAonly.cu ipo_out.o -o cfwccuda_AAonly -L/opt/intel/compilers_and_libraries/mac/lib -limf

cfwccuda_host.o: cfwc.h  cfwccuda_host.c
#	clang++  -Ofast -c cfwccuda_host.cc
#	gcc -O3 -ffast-math -c cfwccuda_host.cc
	icc -fast -ipo-c cfwccuda_host.cc
	mv ipo_out.o cfwccuda_host.o

cfwccuda.o: cfwccuda.c cfwc.h
	nvcc -O3  -use_fast_math -ftz=true -dc cfwccuda.cu

mt19937ar_dev.o: mt19937ar_dev.h mt19937ar_dev.c
	nvcc -O3  -use_fast_math -ftz=true -dc mt19937ar_dev.cu 

ipo_out.o: cfwc_metropolis.f mt19937ar.f
	ifort -fast -ipo-c cfwc_metropolis.f mt19937ar.f

cfwc_metropolis.o: cfwc_metropolis.f
#	dragonegg-3.3-gfortran-mp-4.6 -Ofast -c cfwc_metropolis.f
	gfortran -O3 -ffast-math -c cfwc_metropolis.f


mt19937ar.o: mt19937ar.f
#	dragonegg-3.3-gfortran-mp-4.6 -Ofast -c mt19937ar.f
	gfortran -O3 -ffast-math -c mt19937ar.f
