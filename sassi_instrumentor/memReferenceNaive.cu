#include <cupti.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi_lazyallocator.hpp"

#define WARP_SIZE   32
#define BUFFER_SIZE (WARP_SIZE*WARP_SIZE*WARP_SIZE)

//TO DO: figure out what size this should be
__managed__ intptr_t sassi_references[BUFFER_SIZE];
__managed__ unsigned int memIndex = 0;

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason) {
    FILE *file = fopen("sassi-memReferences.txt", "a");
    fprintf(file, "Memory References:\n");
    for (unsigned i = 0; i <= BUFFER_SIZE; i++) {
        //TO DO: make sure type is right for printing
        fprintf(file, "%p", (void*) sassi_references[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}

static sassi::lazy_allocator referencesInitializer(
    []() {
        //initialize necessary data structures 
        bzero(sassi_references, sizeof(sassi_references));
        // bzero(memIndex, sizeof(memIndex));       
    }, 
    // get the results after kernel execution
    sassi_finalize);


__device__ void sassi_before_handler(SASSIBeforeParams *bp, SASSIMemoryParams *mp) {
    
    if (bp->GetInstrWillExecute()) {
        //only execute if memory operation is a read or write, to be safe
        if (bp->IsMemRead() || bp->IsMemWrite()) { 
            intptr_t mpAddr = mp->GetAddress();
            intptr_t baseAddr = mpAddr & ~0x1FF; // mask the lower 9 bits off 
            unsigned int currentIndex  = atomicAdd(&memIndex, 1);
            sassi_references[currentIndex] = baseAddr;
        }
    }
}
