#include <cupti.h>
#include <stdlib.h>
#include <stdio.h>
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi_lazyallocator.hpp"

#define WARP_SIZE   32
#define BUFFER_SIZE (WARP_SIZE*WARP_SIZE)

//TO DO: figure out what size this should be
__managed__ intptr_t sassi_references[BUFFER_SIZE];
__managed__ unsigned int index;

static sassi::lazy_allocator referencesInitializer(
    []() {
        //initialize necessary data structures 
        bzero(sassi_references, sizeof(sassi_references));
        bzero(index, sizeof(index));
    }, 
    //get the results after kernel execution
    sassi_finalize);

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason) {
    FILE *file = fopen("sassi-memReferences.txt", "a");
    fprintf(file, "Memory References:\n");
    for (unsigned i = 0; i <= BUFFER_SIZE; i++) {
        //TO DO: make sure type is right for printing
        fprintf(file, "%x", sassi_references[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}

__device__ void sassi_before_handler(SASSIBeforeParams *bp, SASSIMemoryParams *mp) {
    
    if (bp->GetInstrWillExecute()) {
        intptr_t mpAddr = mp->GetAddress();
        //only execute if memory operation is a read or write, to be safe
        if (isMemRead() || isMemWrite()) {
            unsigned int currentIndex  = atomicAdd(index, 1);
            sassi_references[currentIndex] = mpAddr;
        }
    }
}
