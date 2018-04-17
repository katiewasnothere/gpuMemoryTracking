#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <lynx/instrumentation/interface/PTXInstrumentor.h>

namespace instrumentation {
    class MemoryReferencesInstrumentor : public PTXInstrumentor {
        
        public:

            size_t *buffer;
            unsigned int size;
            unsigned int index;
        public:

            MemoryReferencesInstrumentor();

            bool validate();

            void analyze(ir::Module &module);

            void initialize();

            std::string specificationPath();

            void extractResults(std::ostream *out);            
    
    };

    MemoryReferencesInstrumentor::MemoryReferencesInstrumentor() :
    buffer(0), size(0), index(0) {

    }
        
    bool MemoryReferencesInstrumentor::validate() {
        return true;
    }

    void MemoryReferencesInstrumentor::analyze(ir::Module &module) {
        //TODO: decide if we want to analyze anything
    }

    void MemoryReferencesInstrumentor::initialize() {
        //TODO: fix and add where necessary
        
        // variables for device code
        buffer = 0;
        index = 0;

        // variable for host code
        //TODO: figure out what size will be
        size = 500; 

        if (cudaMalloc((void**) &buffer, size * sizeof(size_t)) != cudaSuccess) {
                std::cerr << "cudaMalloc failed for buffer allocation" << std::endl;
                exit(-1);
        }

        if (cudaMalloc((void**) &index, sizeof(unsigned int*)) != cudaSuccess) {
            std::cerr << "cudaMalloc failed for index symbol allocation" << std::endl;
            exit(-1);
        }

        if (cudaMemset(buffer, 0, size * sizeof(size_t)) != cudaSuccess) {
            std::cerr << "cudaMemset failed for buffer setting" << std::endl;
            exit(-1);
        }

        if (cudaMemcpyToSymbol(GLOBAL_MEM_BASE_ADDRESS, &buffer, sizeof(size_t*), 0, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "cudaMemcpytoSymbol failed for global memory address of buffer" << std::endl;
            exit(-1);
        }
        
    }

    std::string MemoryReferencesInstrumentor::specificationPath() {
        std::string path =  "../instrumentation/measureMemoryReferences.c";  
        return path;
    }

    void MemoryReferencesInstrumentor::extractResults(std::ostream *out) {
        //TODO: can't store mem references until count determined
        size_t* info = new size_t[size];

        if (buffer) {
            cudaMemcpy(info, buffer, size * sizeof(size_t), cudaMemcpyDeviceToHost);
            cudaFree(buffer);
        }
    }


}
