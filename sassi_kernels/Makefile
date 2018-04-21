EXTRA_LINK_FLAGS = -Xlinker -rpath,/usr/local/sassi7/extras/CUPTI/lib64

all: vectorMath

vectorMath.o: vectorMath.cu
	/usr/local/sassi7/bin/nvcc -D_FORCE_INLINES -c -O3	\
	   		-gencode arch=compute_50,code=sm_50 	\
	      	-Xptxas --sassi-inst-before='memory' 	\
		    -Xptxas --sassi-before-args='mem-info' 	\
			-Xptxas --sassi-debug					\
			-dc 									\
			-o vectorMath.o vectorMath.cu

vectorMath: vectorMath.o
	/usr/local/sassi7/bin/nvcc -D_FORCE_INLINES			\
		-o vectorMath vectorMath.o 						\
		$(EXTRA_LINK_FLAGS) 							\
		-gencode arch=compute_50,code=sm_50 -lcudadevrt	\
		-L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti 	\
		-L../sassi_instrumentor/lib -lmemReferenceNaive

clean:
	rm vectorMath.o	
