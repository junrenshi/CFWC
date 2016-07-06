
cfwccuda: cfwccuda.c tinymt32_kernel.c cfwccuda_host.o ipo_out.o
	nvcc -O3  cfwccuda.cu cfwccuda_host.o ipo_out.o -o cfwccuda -limf -lintlc -lsvml

cfwccuda_host.o: cfwc.h  cfwccuda_host.c
	icc -fast -ipo-c cfwccuda_host.cc
	mv ipo_out.o cfwccuda_host.o

ipo_out.o: cfwc_metropolis.f mt19937ar.f
	ifort -fast -ipo-c cfwc_metropolis.f mt19937ar.f
