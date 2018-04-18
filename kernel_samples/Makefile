INSTRUMENTOR_PATH := ../instrumentor/

all: MemoryReferencesInstrumentor

MemoryReferencesInstrumentor: MemoryReferencesInstrumentor.o vectorMath.o
	g++ *.o -o MemoryReferencesInstrumentor -L/usr/local/lib -lcudart  -llynx_runtime -L/usr/local/cuda/lib64

MemoryReferencesInstrumentor.o: $(INSTRUMENTOR_PATH)MemoryReferencesInstrumentor.cpp
	g++ -std=c++0x $(INSTRUMENTOR_PATH)MemoryReferencesInstrumentor.cpp -c -I/usr/local/include -I/usr/local/cuda/include

vectorMath.o: vectorMath.cu
	nvcc -arch=sm_30 -c vectorMath.cu

clean: 
	rm -rf *o MemoryReferencesInstrumentor
