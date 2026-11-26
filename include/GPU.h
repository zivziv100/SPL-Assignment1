#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include <unordered_map>

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;

    PrefixMatch() : matchedTokens(0), blockIds() {}
};

class MemoryBlock;
class GenerativeRequest;
class EmbeddingRequest;
class Request;

class GPU{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    ~GPU();

    GPU(const GPU& other);
    GPU& operator=(const GPU& other);

    GPU(GPU&& other) noexcept;
    GPU& operator=(GPU&& other) noexcept;
    
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;
    std::size_t getBlockSize() const; // Added Getter for blockSize for Requests
    int getCacheHits() const;
    int getEvictions() const;
    void acquireCached(std::size_t slot);

private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::unordered_map<int, std::size_t> reqIdToBlocks;
    std::size_t totalReserved;
    std::vector<std::string> modelIds;
    std::vector<std::vector<std::string>> prefixKeys;
    int cacheHits;
    int evictions;

    void removeBlock(std::size_t slot);
    void clear();
    void copyBlocksFrom(const GPU& other);
    std::size_t findSlotWithKey(const std::string& modelId, const std::vector<std::string>& key) const;
    std::size_t findSlotForNewBlock(const std::vector<std::size_t>& slot);
};

#endif