#include <cupti.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sassi/sassi-core.hpp>
#include <sassi/sassi-memory.hpp>
#include "sassi_lazyallocator.hpp"
#include "sassi_intrinsics.h"

#define WARP_SIZE   32
#define BUFFER_SIZE (WARP_SIZE*WARP_SIZE)

//NOTE: this size must be changed depending on the program being used
__managed__ intptr_t sassiReferences[BUFFER_SIZE];
__device__ unsigned int memIndex = 0;

static void sassi_finalize(sassi::lazy_allocator::device_reset_reason reason) {
    FILE *file = fopen("sassi-memReferencesCoalesce.txt", "a");
    fprintf(file, "Memory References:\n");

    for (unsigned i = 0; i <= BUFFER_SIZE; i++) {
        fprintf(file, "%p\n", (void*) sassiReferences[i]);
    }
    fprintf(file, "\n");
    fclose(file);
}

static sassi::lazy_allocator referencesInitializer(
    []() {
        //initialize necessary data structures 
        bzero(sassiReferences, sizeof(sassiReferences));
    }, 
    // get the results after kernel execution
    sassi_finalize);

__device__ void sassi_before_handler(SASSIBeforeParams *bp, SASSIMemoryParams *mp) {
    if (bp->GetInstrWillExecute()) {
        //only execute if memory operation is a read or write, to be safe
        if (bp->IsMemRead() || bp->IsMemWrite()) { 
            intptr_t baseAddr = mp->GetAddress() & ~0x1FF;
            
            int availableThreads = __ballot(1);
            int threadLaneId = get_laneid();
            intptr_t leadersBaseAddr = NULL;
            int shuffleLeader = 0;
            int matchedThreads = 0;

Loop:
            // check conditions 
            if (availableThreads == 0 ){
                assert(baseAddr != 0);
                return;
            }
            // choose a leader and broadcast it's base address
            shuffleLeader = __ffs(availableThreads) - 1;
            leadersBaseAddr = __broadcast<intptr_t>(baseAddr, shuffleLeader);
                
            // find the threads in the warp with a matching base address
            matchedThreads = __ballot(leadersBaseAddr == baseAddr);

            // update the number of threads who's address needs to be checked
            availableThreads = availableThreads & ~matchedThreads;   
               
           // if this thread is the leader, add the base addr to global array 
            if (threadLaneId == shuffleLeader) {
                int currentIndex  = atomicAdd(&memIndex, 1);
                assert(currentIndex >= 0);
                // will only ever access an index once, so this is safe
                sassiReferences[currentIndex] = baseAddr;
            }
            goto Loop;
        }
    }
}
