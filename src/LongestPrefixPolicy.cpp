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

size_t LongestPrefixPolicy::maxCachedPrefix(const Request* req, const vector<GPU>& gpus) {
    size_t h = 0;
    for (size_t i = 0; i < gpus.size(); i++) {
        h = max(h, req->getCachedPrefix(gpus[i]).matchedTokens);
    }
    return h;
}

bool LongestPrefixPolicy::comparePrefix (const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus, int a, int b) {
    size_t maxPrefixA = maxCachedPrefix(eligible[a], gpus);
    // חישוב h עבור בקשה b (המקסימום על פני כל ה-GPUs)
    size_t maxPrefixB = maxCachedPrefix(eligible[b], gpus);;

    if (maxPrefixA != maxPrefixB) {
        return maxPrefixA > maxPrefixB;
    }

    return eligible[a]->getId() < eligible[b]->getId();
}
   