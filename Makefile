
cfwccuda: cfwccuda.cu tinymt32_kernel.cuh cfwccuda_host.o ipo_out.o
	nvcc -O3  cfwccuda.cu cfwccuda_host.o ipo_out.o -o cfwccuda -limf -lintlc -lsvml

cfwccuda.cu:
	ln -s -f cfwccuda.c cfwccuda.cu

tinymt32_kernel.cuh:
	ln -s -f tinymt32_kernel.c tinymt32_kernel.cuh

cfwccuda_host.o: cfwc.h  cfwccuda_host.cc
	icc -fast -ipo-c cfwccuda_host.cc
	mv ipo_out.o cfwccuda_host.o

cfwccuda_host.cc:
	ln -s -f cfwccuda_host.c cfwccuda_host.cc

ipo_out.o: cfwc_metropolis.f mt19937ar.f
	ifort -fast -ipo-c cfwc_metropolis.f mt19937ar.f
