EXTRA_LINK_FLAGS = -Xlinker -rpath,/usr/local/sassi7/extras/CUPTI/lib64

SASSI_FLAGS = -Xptxas --sassi-inst-before='memory'  -Xptxas --sassi-before-args='mem-info'  -Xptxas --sassi-debug	-Xptxas --sassi-iff-true-predicate-handler-call

GEN_CODE = -gencode arch=compute_50,code=sm_50 

CUPTI_INC = -L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti

all: coalesce

vectorMath.o: vectorMath.cu
	/usr/local/sassi7/bin/nvcc -D_FORCE_INLINES -c -O3 \
			$(GEN_CODE)								\
			$(SASSI_FLAGS)							\
			-dc 									\
			-o vectorMath.o vectorMath.cu

naive: vectorMath.o
	/usr/local/sassi7/bin/nvcc -D_FORCE_INLINES	-O3		\
		-o vectorMathNaive vectorMath.o 				\
		$(EXTRA_LINK_FLAGS) 							\
		$(GEN_CODE)										\
		$(CUPTI_INC)									\
		-L../sassi_instrumentor/lib -lmemReferenceNaive

coalesce: vectorMath.o
	/usr/local/sassi7/bin/nvcc -D_FORCE_INLINES	-O3		\
		-o vectorMathCoalesce vectorMath.o 						\
		$(EXTRA_LINK_FLAGS) 							\
		$(GEN_CODE)										\
		$(CUPTI_INC)									\
		-L../sassi_instrumentor/lib -lmemReferenceCoalesce

clean:
	rm -f vectorMath.o
	rm -f vectorMathNaive 
	rm -f vectorMathCoalesce

r: clean all
