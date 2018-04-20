#Makefile for using vector addition kernel with cuda SASSI

NVCC = /usr/local/cuda-7.0/bin/nvcc
CUPTI = /usr/local/cuda-7.0/extras/CUPTI
SASSI_PATH = /usr/local/sassi7
GEN_CODES = -gencode arch=compute_35, code=sm+35
CC_FLAGS = -std=c++11 -W -O3

naive: memReferenceNaive.cu
	$(NVCC) $(CC_FLAGS) $(GEN_CODES) -rdc=true -o memReferenceNaive \
					--maxrregcount=16 						\
					-I$(SASSI_PATH)/include 				\ 
					-I$(CUPTI)/include 						\ 

