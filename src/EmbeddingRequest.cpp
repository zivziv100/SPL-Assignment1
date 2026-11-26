#include <Model.h>
#include <GPU.h>
#include "Request.h"

using namespace std;

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt) : Request(id, model, admissionCycle, prompt), currentTokenIndex(0), result() {}
EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest& other, const Model& model) : Request(other, model) , currentTokenIndex(other.currentTokenIndex), result(other.result) {}
EmbeddingRequest::~EmbeddingRequest() { }

size_t EmbeddingRequest::getMaxNeededMemory() const{
    return model.getEmbeddingSize() * 2;
}

size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
    return getMaxNeededMemory(); // the needed memmory is the max needed memory, which is 2 * embeddingSize
}

vector<string> EmbeddingRequest::getCacheableTokens() const {
    return vector<string>();
}

string EmbeddingRequest::getOutput() const {
    string stringResult = "[";
    for (size_t i = 0; i < result.size(); i++) {
        if (i != 0) { stringResult += ", "; } // Add "," before every number, except the first (i=0)
        stringResult += to_string(result[i]);
    }
    stringResult += "]";
    return stringResult;
}

size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks; // represents the reservedBlocks
}

bool EmbeddingRequest::finished() const {
    return currentTokenIndex == tokens.size(); // if we got to the end of the tokens, we done
}

Request* EmbeddingRequest::clone(const Model& model) const {
    return new EmbeddingRequest(*this, model); // using the Copy-Constructor
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot); // using the Double-Dispatch
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    size_t neededBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu)); // getting the needed blocks for the request
    reservedBlocks = neededBlocks; // "reserving" them
    gpu.reserve(neededBlocks, *this);

    for (size_t i = 0; i < neededBlocks; i++) { // allocating each block, therefore 'reservedBlocks' decreases
        size_t slot = gpu.allocateBlock(*this); 
        reservedBlocks = reservedBlocks - 1;
        memoryBlocks.push_back(slot);
    }
}

void EmbeddingRequest::act(GPU& gpu) {
    size_t e = model.getEmbeddingSize();
    vector<string> temp;
    temp.push_back(tokens[currentTokenIndex]); // taking the current token

    model.generateTokenEmbedding(temp, gpu, getSlot(e, gpu), getOffset(e, gpu));

    for (size_t i = 0; i < e; i++) {
        int iSum = gpu.readFromBlock(getSlot(i, gpu), getOffset(i, gpu));
        int currentValue = gpu.readFromBlock(getSlot(i+e, gpu), getOffset(i+e, gpu));
        gpu.writeToBlock(getSlot(i, gpu), getOffset(i, gpu), iSum + currentValue);
    }

    currentTokenIndex++;

    if (finished()) {
        int tokensNumber = static_cast<int>(tokens.size());
        for (size_t j = 0; j < e; j++) {
            result.push_back(gpu.readFromBlock(getSlot(j, gpu), getOffset(j, gpu)) / tokensNumber); // Calculating the mean (avg)
        }
    }
}

size_t EmbeddingRequest::getSlot(size_t k, const GPU& gpu) const {
    return memoryBlocks[k / gpu.getBlockSize()];
}

size_t EmbeddingRequest::getOffset(size_t k, const GPU& gpu) const {
    return k % gpu.getBlockSize();
}
