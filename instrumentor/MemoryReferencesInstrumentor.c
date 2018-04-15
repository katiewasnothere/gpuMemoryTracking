#include <stdlib.h>

namespace instrumentation{
    class MemoryReferencesInstrumentor : public PTXInstrumentor {
        
        public:

            MemoryReferencesInstrumentor();

            bool validate();

            void analyze(ir::Module &module);

            void initialize();

            std::string specificationPath();

            void extractResults(std::ostream *out);            
    
    };
    
    //TODO: make constructor
    
    bool MemoryReferencesInstrumentor::validate() {
        return true;
    }

    void MemoryReferencesInstrumentor::analyze(ir::Module &module) {
        //TODO: decide if we want to analyze anything
    }

    void MemoryReferencesInstrumentor::initialize() {
        //TODO: implement
    }

    void MemoryReferencesInstrumentor::specificationPath() {
           return "../instrumentation/measureMemoryReferences.c";
    }

    void MemoryReferencesInstrumentor::extractResults(std::ostream *out) {
        //TODO: implement
    }


}
