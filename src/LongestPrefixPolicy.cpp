#include <SchedulingPolicy.h>
#include "Request.h"
#include "MemoryBlock.h"
#include "Model.h"
#include "GPU.h"
#include <algorithm>
#include <stdexcept>  
using namespace std;


SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}


vector<int> LongestPrefixPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
 vector<int> orderOfIndices;
    for (size_t i = 0; i < eligible.size(); i++) {
        orderOfIndices.push_back(static_cast<int>(i));
    }
    std::sort(orderOfIndices.begin(), orderOfIndices.end(), [&eligible, &gpus](int a, int b) {
        return comparePrefix(eligible, gpus, a, b);
    });

    return orderOfIndices;
}

bool LongestPrefixPolicy::comparePrefix (const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus, int a, int b) {
   
    if (a<0 || a>=eligible.size() || b<0 || b>=eligible.size()) {
        throw out_of_range("Index is out of range of eligible requests");
    }

size_t maxPrefixA = 0;
    vector<string> tokensA = eligible[a]->getCacheableTokens();
    if (!tokensA.empty()) {
        const Model& modelA = eligible[a]->getModel();
        for (size_t i = 0; i < gpus.size(); i++) {
            PrefixMatch match = gpus[i].findLongestCachedPrefix(modelA.getModelId(), modelA.getEmbeddingSize(), tokensA);
            if (match.matchedTokens > maxPrefixA) {
                maxPrefixA = match.matchedTokens;
            }
        }
    }

    // חישוב h עבור בקשה b (המקסימום על פני כל ה-GPUs)
    size_t maxPrefixB = 0;
    vector<string> tokensB = eligible[b]->getCacheableTokens();
    if (!tokensB.empty()) {
        const Model& modelB = eligible[b]->getModel();
        for (size_t i = 0; i < gpus.size(); i++) {
            PrefixMatch match = gpus[i].findLongestCachedPrefix(modelB.getModelId(), modelB.getEmbeddingSize(), tokensB);
            if (match.matchedTokens > maxPrefixB) {
                maxPrefixB = match.matchedTokens;
            }
        }
    }

    if (maxPrefixA != maxPrefixB) {
        return maxPrefixA > maxPrefixB;
    }

    return eligible[a]->getId() < eligible[b]->getId();


}
   