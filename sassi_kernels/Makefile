all: vectorMath

vectorMath.o: vectorMath.cu
	/usr/local/sassi7/bin/nvcc -I./inc -c -O3 		\
	   		-gencode arch=compute_50,code=sm_50 	\
	      	-Xptxas --sassi-inst-after='memory'		\
		    -Xptxas --sassi-after-args='mem-info' 	\
			-dc 									\
			-o vectorMath.o vectorMath.cu

vectorMath: vectorMath.o
	/usr/local/sassi7/bin/nvcc -o vectorMath vectorMath.o 	\
		-gencode arch=compute_50,code=sm_50 				\
		-lcudadevrt 										\
		-L/usr/local/sassi7/extras/CUPTI/lib64 -lcupti 		\
		-L../instlibs/lib -lvalueprof

clean:
	rm vectorMath
