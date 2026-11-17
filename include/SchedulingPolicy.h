#ifndef SCHEDULING_POLICY_H
#define SCHEDULING_POLICY_H

class GPU;
class Request;

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
};

class ArrivalOrderPolicy : public SchedulingPolicy { 
private:
    static bool compareByOrder(const Request* a, const Request* b);
public:
    virtual ~ArrivalOrderPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const;
    virtual SchedulingPolicy* clone() const;
};
class ShortestJobPolicy : public SchedulingPolicy { 

};
class LongestPrefixPolicy : public SchedulingPolicy {

};
#endif