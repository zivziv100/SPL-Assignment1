#include <Model.h>
#include <GPU.h>
#include "Request.h"

using namespace std;

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens) : Request(id, model, admissionCycle, prompt) , 
    scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens), tokensOutput(model.tokenizer(scriptedOutput)), tokensInMemory(0) 
    {
        if (tokensOutput.size() > maxOutputTokens) {
            tokensOutput.resize(maxOutputTokens);
        }
    }
GenerativeRequest::GenerativeRequest(const GenerativeRequest& other, const Model& model) : Request(other, model), 
scriptedOutput(other.scriptedOutput), maxOutputTokens(other.maxOutputTokens), tokensOutput(other.tokensOutput), tokensInMemory(other.tokensInMemory) {} 

size_t GenerativeRequest::getMaxNeededMemory() const{
    size_t n = tokens.size();
    size_t m = maxOutputTokens;
    size_t embeddingSize = model.getEmbeddingSize();
    return embeddingSize * (m + n);
}

size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    size_t n = tokens.size();
    size_t m = maxOutputTokens;
    size_t embeddingSize = model.getEmbeddingSize();
    PrefixMatch prefix = gpu.findLongestCachedPrefix(model.getModelId(), embeddingSize, tokens);
    size_t h = prefix.matchedTokens;
    return embeddingSize * (n - h + m);
}

vector<string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    PrefixMatch prefix = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
    for (size_t i = 0; i < prefix.blockIds.size(); i++) {
        gpu.acquireCached(prefix.blockIds[i]);
        memoryBlocks.push_back(prefix.blockIds[i]);
    }
    tokensInMemory = prefix.matchedTokens;

    size_t neededBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    reservedBlocks = neededBlocks;
    gpu.reserve(neededBlocks, *this);
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) return;
    
    size_t e = model.getEmbeddingSize();
    size_t i = tokensInMemory;
    size_t t = gpu.getBlockSize() / e;

    if (i % t == 0) {
        size_t newSlot = gpu.allocateBlock(*this);
        reservedBlocks--;
        memoryBlocks.push_back(newSlot);
    }

    size_t slot = memoryBlocks[i / t];
    vector<string> prefix = getSequencePrefix(i + 1);
    model.generateTokenEmbedding(prefix, gpu, slot, (i % t) * e);
    tokensInMemory++;

    if (tokensInMemory % t == 0) {
        gpu.assignPrefixToBlock(slot, prefix, model.getModelId());
    }
}

vector<string> GenerativeRequest::getSequencePrefix(size_t length) const {
    vector<string> result;
    result.reserve(length);
    for (size_t i = 0; i < length; i++) {
        if (i < tokens.size()) result.push_back(tokens[i]);
        else result.push_back(tokensOutput[i - tokens.size()]);
    }
    return result;
}


bool GenerativeRequest::finished() const {
    return tokensInMemory == (tokens.size() + tokensOutput.size());
}

string GenerativeRequest::getOutput() const {
    string result;
    for (size_t i = 0; i < tokensOutput.size(); i++) {
        if (i != 0) result += " ";
        result += tokensOutput[i];
    }
    return result;
}

Request* GenerativeRequest::clone(const Model& model) const {
    return new GenerativeRequest(*this, model);
}
