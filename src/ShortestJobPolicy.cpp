#include <SchedulingPolicy.h>
#include "Request.h"
#include "MemoryBlock.h"
#include "GPU.h"
#include <algorithm>
#include <stdexcept>   


using namespace std;


SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

vector<int> ShortestJobPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
 vector<int> orderOfIndices;
    for (size_t i = 0; i < eligible.size(); i++) {
        orderOfIndices.push_back(static_cast<int>(i));
    }
    std::sort(orderOfIndices.begin(), orderOfIndices.end(), [&eligible, &gpus](int a, int b) {
        return compareIndices(eligible, gpus, a, b);
    });

    return orderOfIndices;
}



   bool compareIndices (const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus, int a, int b) {
   
    if (a<0 || a>=eligible.size() || b<0 || b>=eligible.size()) {
        throw out_of_range("Index is out of range of eligible requests");
    }
   
    size_t blocks1 = gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory());
    size_t blocks2 = gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory());

    if (blocks1 != blocks2) {
        return blocks1 < blocks2;
    }
    return eligible[a]->getId() < eligible[b]->getId();
}

    
