#include <SchedulingPolicy.h>
#include <Request.h>
#include <algorithm>
using namespace std;

vector<int> ArrivalOrderPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
    vector<int> order;
    for (size_t i = 0; i < eligible.size(); i++) {
        order.push_back(static_cast<int>(i));
    }
    std::sort(order.begin(), order.end(), [&eligible](int a, int b) {
        return compareByOrder(eligible[a], eligible[b]);
    });
    return order;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

bool ArrivalOrderPolicy::compareByOrder(const Request* a, const Request* b){
    int aId = a -> getId();
    int bId = b -> getId();

    int aCycle = a -> getAdmissionCycle();
    int bCycle = b -> getAdmissionCycle();

    if (aCycle < bCycle) return true;
    else if (bCycle < aCycle) return false;
    else return aId < bId;
}