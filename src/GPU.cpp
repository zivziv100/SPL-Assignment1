#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
using namespace std;

GPU::GPU(size_t blocksNum, size_t blockSize) : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr) {}

size_t GPU::getBlocksForMemory(size_t memory) const {
    return (memory +blockSize - 1) / blockSize;
} 
void GPU::reserve(size_t n, const Request& request) {
    for (size_t i = 0; i < n; i++) {
        size_t slot = allocateBlock(request);
        if (memory[slot] != nullptr) {
            memory[slot]->acquire();
        }
    }
}

void GPU::unreserve(size_t n, const Request& request) {
    
}

