#include <Model.h>
#include <GPU.h>
#include "Request.h"

using namespace std;

// Rule Of 3
Request::Request(int id, const Model& model, int admissionCycle, std::string prompt) : id(id) , model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0) {}
Request::Request(const Request& other, const Model& model) : id(other.id), model (model), admissionCycle(other.admissionCycle), tokens(other.tokens), memoryBlocks(other.memoryBlocks), reservedBlocks(other.reservedBlocks) {}
Request::~Request() {}


// Getters
int Request:: getId() const { return id; }
const Model& Request::getModel() const { return model; }
int Request::getAdmissionCycle() const { return admissionCycle; }

void Request::deallocateMemory(GPU& gpu) {
    size_t needToUnreserve = getNeedToUnreserve();
    gpu.unreserve(needToUnreserve, *this);
    reservedBlocks = 0;

    for (size_t i = 0; i < memoryBlocks.size(); i++) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
}