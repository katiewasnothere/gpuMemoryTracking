#Makefile for using vector addition kernel with cuda SASSI

NVCC = /usr/local/cuda-7.0/bin/nvcc
CUPTI = /usr/local/cuda-7.0/extras/CUPTI
SASSI_PATH = /usr/local/sassi7
GEN_CODES = -gencode arch=compute_50,code=sm_50
CC_FLAGS = -std=c++11 -O3 --compiler-options -Wall

export LIBS := 	libmemReferenceNaive.a	  \
				libmemReferenceCoalesce.a \

INSTALLED_LIBS := $(addprefix ./lib/, $(LIBS))

install: $(INSTALLED_LIBS) 

./lib/lib%.a: %.o
	ar r $@ $^
	ranlib $@

%.o: %.cu
	$(NVCC) -D_FORCE_INLINES  $(CC_FLAGS) 	\
		$(GEN_CODES)             			\
	    -lineinfo -c -rdc=true -o $@ $^   	\
		--maxrregcount=16          			\
		--compiler-options -Wall			\
		-I$(SASSI_PATH)/include    			\
		-I$(CUPTI)/include/        			\
		-I$(SASSI_PATH)/include/sassi 		\
		-I./utils
