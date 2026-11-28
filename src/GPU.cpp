#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <algorithm>
#include <utility>

using namespace std;

// ----------- RULE OF 5 : START -------------
GPU::GPU(size_t blocksNum, size_t blockSize) : blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), reqIdToBlocks(), totalReserved(0), modelIds(blocksNum), prefixKeys(blocksNum), cacheHits(0), evictions(0) {}
GPU::~GPU() {
    clear();
}

GPU::GPU(const GPU& other) : blocksNum(other.blocksNum) , blockSize(other.blockSize), memory(other.memory.size(), nullptr), reqIdToBlocks(other.reqIdToBlocks), totalReserved(other.totalReserved), modelIds(other.modelIds), prefixKeys(other.prefixKeys), cacheHits(other.cacheHits), evictions(other.evictions) {
    copyBlocksFrom(other);
}
GPU& GPU::operator=(const GPU& other) {
    if (this != &other) {
        this->clear();
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        reqIdToBlocks = other.reqIdToBlocks;
        modelIds = other.modelIds;
        prefixKeys = other.prefixKeys;
        cacheHits = other.cacheHits;
        evictions = other.evictions;
        totalReserved = other.totalReserved;

        memory.assign(other.memory.size(), nullptr);
        copyBlocksFrom(other);
    }
    return *this;
}

GPU::GPU(GPU&& other) noexcept
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(std::move(other.memory)),
      reqIdToBlocks(std::move(other.reqIdToBlocks)),
      totalReserved(other.totalReserved),
      modelIds(std::move(other.modelIds)),
      prefixKeys(std::move(other.prefixKeys)),
      cacheHits(other.cacheHits), 
      evictions(other.evictions) {
    other.memory.clear();
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
}
GPU& GPU::operator=(GPU&& other) noexcept {
    if (this != &other) {
        clear();

        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory = std::move(other.memory);
        reqIdToBlocks = std::move(other.reqIdToBlocks);
        totalReserved = other.totalReserved;
        modelIds = std::move(other.modelIds);
        prefixKeys = std::move(other.prefixKeys);
        cacheHits = other.cacheHits;
        evictions = other.evictions;

        other.memory.clear();
        other.blocksNum = 0;
        other.blockSize = 0;
        other.totalReserved = 0;
        other.cacheHits = 0;
        other.evictions = 0;
    }
    return *this;
}

void GPU::copyBlocksFrom(const GPU& other) {
    for (size_t i = 0; i < other.memory.size(); i++) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            int ref = other.memory[i]->getRefCount();
            for (int j = 0; j < ref; j++) {
                memory[i]->acquire();
            }
        }
    }
}
void GPU::clear() {
    for (size_t i = 0; i < memory.size(); i++) {
        delete memory[i];
        memory[i] = nullptr;
    }
} 
// ----------- RULE OF 5 : END -------------

size_t GPU::getBlocksForMemory(size_t memory) const {
    return (memory +blockSize - 1) / blockSize;
}

void GPU::reserve(size_t n, const Request& request) {
    int reqId = request.getId();
    size_t& reqBlocks = reqIdToBlocks[reqId];

    reqBlocks += n;
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

void GPU::acquire(size_t slot) {
    memory[slot] -> acquire();
}
void GPU::acquireCached(size_t slot) {
    this->acquire(slot);
    cacheHits++;
}

int GPU::readFromBlock(size_t blockId, size_t offset) const {
    return memory[blockId] -> readFromBlock(offset);
}
void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    if (blockId >= blocksNum || memory[blockId] == nullptr) {
        return;
    }
    memory[blockId]->writeToBlock(offset, value);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
}
void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    (void)request; 
    memory[slot] = new KVBlock(blockSize);
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

PrefixMatch GPU::findLongestCachedPrefix(const string& modelId, size_t embeddingSize,const vector<string>& promptTokens) const {
    PrefixMatch result;
    result.matchedTokens = 0;

    size_t t = blockSize / embeddingSize;
    vector<string> prefix;

    size_t start = prefix.size();
    while (prefix.size() + t <= promptTokens.size()) {
        for (size_t i = 0; i < t; i++) {
            prefix.push_back(promptTokens[start]);
            start++;
        }
        bool found = false;
        size_t slot = findSlotWithKey(modelId, prefix);
        if (slot != blocksNum) {
            found = true;
            result.blockIds.push_back(slot);
            result.matchedTokens =  prefix.size();
        }
        if (!found) break;
    }
    return result;
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

void GPU::copyMemoryFromOtherGPU(const GPU& other, const vector<std::size_t>& blocks) {
    vector<size_t> protectedSlots;
    for (size_t i = 0; i < blocks.size(); i++) {
        size_t s = blocks[i];
        const string& modelId = other.modelIds[s];
        const vector<string>& prefixKey = other.prefixKeys[s];

        size_t existing = this->findSlotWithKey(modelId, prefixKey);
        if (existing != blocksNum) {
            protectedSlots.push_back(existing);
            continue;
        }

        size_t target = findSlotForNewBlock(protectedSlots);
        memory[target] = other.memory[s]->clone();
        modelIds[target] = modelId;
        prefixKeys[target] = prefixKey;
        protectedSlots.push_back(target);
    }
}

void GPU::removeBlock(size_t slot) {
    delete memory[slot];
    memory[slot] = nullptr;

    modelIds[slot].clear();
    prefixKeys[slot].clear();
}

size_t GPU::allocateBlock(const Request& request) {
    int reqId = request.getId();
    size_t res = reqIdToBlocks.count(reqId);
    if (res == 0 || reqIdToBlocks.at(reqId) == 0) {
        return blocksNum;
    }

    size_t slot = findSlotForNewBlock(vector<size_t>());
    request.allocateOneBlock(*this, slot);
    acquire(slot);

    reqIdToBlocks.at(reqId)--;
    totalReserved--;

    return slot;
}

size_t GPU::findSlotWithKey(const string& modelId, const vector<string>& key) const {
    for (size_t i = 0; i < memory.size(); i++) {
        if (memory[i] != nullptr && modelIds[i] == modelId && prefixKeys[i] == key) {
            return i;
        }
    }
    return blocksNum;
}
size_t GPU::findSlotForNewBlock(const vector<size_t>& slots) {
    for (size_t i = 0; i < memory.size(); i++) {
        if (memory[i] == nullptr) return i;
    }

    for (size_t i = 0; i < memory.size(); i++) {
        if (memory[i] -> getRefCount() == 0 && find(slots.begin(), slots.end(), i) == slots.end()) {
            removeBlock(i);
            evictions++;
            return i;
        }
    }
    return blocksNum;
}

bool GPU::isIdle(size_t slot) const {
    return memory[slot] != nullptr && memory[slot]->getRefCount() == 0;
}

// ----------- Getters : START -------------
int GPU::getCacheHits() const { return cacheHits; }
int GPU::getEvictions() const { return evictions; }
size_t GPU::getBlockSize() const { return blockSize; }
size_t GPU::getBlocksNum() const { return blocksNum; }
// ----------- Getters : END -------------