#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
using namespace std;

GPU::GPU(size_t blocksNum, size_t blockSize) : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr) {}

size_t GPU::getBlocksForMemory(size_t memory) const {
    return (memory +blockSize - 1) / blockSize;
} 
void GPU::reserve(size_t n, const Request& request) {
   if (n == 0) {
        return;
    }
   reqIdToBlocks[request.getId()] += n;
    totalReserved += n;
}

void GPU::unreserve(size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    int reqId = request.getId();
    auto it = reqIdToBlocks.find(reqId);
    if (it != reqIdToBlocks.end()) {
        std::size_t toRemove = (n > it->second) ? it->second : n;
        it->second -= toRemove;
        totalReserved = (toRemove > totalReserved) ? 0 : (totalReserved - toRemove);
        if (it->second == 0) {
            reqIdToBlocks.erase(it);
        }
    }
}

std::size_t GPU::availableSlots() const {
    std::size_t activeBlocks = 0;
    for (const MemoryBlock* block : memory) {
        if (block != nullptr && block->getRefCount() > 0) {
            ++activeBlocks;
        }
    }

    std::size_t usedOrReserved = activeBlocks + totalReserved;
    if (usedOrReserved >= memory.size()) {
        return 0;
    }
    return memory.size() - usedOrReserved;
}
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    if (memory == 0 || blockSize == 0) {
        return 0;
    }
    return (memory + blockSize - 1) / blockSize;
}
// Double - Dispatch!!
void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    (void)request; 
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    (void)request; 
    memory[slot] = new EmbeddingWorkspace(blockSize);
}

void GPU::release(std::size_t slot) {
    if (slot >= blocksNum || memory[slot] == nullptr) {
        return;
    }

    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() == 0) {
        bool hasPrefixKey = !prefixKeys[slot].empty();
        if (action == IdleAction::Destroy || !hasPrefixKey) {
            removeBlock(slot);
        }
    }
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    if (blockId >= blocksNum || memory[blockId] == nullptr) {
        return;
    }
    memory[blockId]->writeToBlock(offset, value);
}


void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    if (block >= blocksNum || memory[block] == nullptr || prefixTokens.empty()) {
        return;
    }

    for (std::size_t i = 0; i < blocksNum; ++i) {
        if (i != block && memory[i] != nullptr &&
            modelIds[i] == modelId && prefixKeys[i] == prefixTokens) {
            return;
        }
    }

    modelIds[block] = modelId;
    prefixKeys[block] = prefixTokens;
}
