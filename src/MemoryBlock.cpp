#include "MemoryBlock.h"
#include <algorithm>   
#include <stdexcept>   

// --------- RULE OF 5 : START ------------
MemoryBlock::MemoryBlock(size_t size) : size(size), refCount(0), data(new int[size])  
{
    for (size_t i = 0; i < size; i++) {
        data[i] = 0;
    }
}

MemoryBlock::~MemoryBlock() { delete[] data; }

MemoryBlock::MemoryBlock(const MemoryBlock& other) : size(other.size), refCount(0), data(new int[other.size]) {
    for (size_t i = 0; i < size; i++) {
        data[i] = other.data[i];
    }
}
MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        int* newData = new int[other.size];
        for (size_t i = 0; i < other.size; i++) {
            newData[i] = other.data[i];
        }
        delete[] data;

        data = newData;
        size = other.size;
        refCount = 0;
    }
    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) : size(other.size), refCount(other.refCount), data(other.data) {
    other.data = nullptr;
    other.size = 0;
    other.refCount = 0;
}
MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    if (this != &other) {
        delete[] data;

        data = other.data;
        size = other.size;
        refCount = other.refCount;

        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }
    return *this;
}
// --------- RULE OF 5 : END --------------

int MemoryBlock::readFromBlock(std::size_t offset) const {
    if (offset >= size) {
        throw std::out_of_range("Offset is not in the range of the block size");
    }
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    if (offset >= size) {
        throw std::out_of_range("Offset is not in the range of the block size");
    }
    data[offset] = value;
}

void MemoryBlock::acquire() {
    refCount++;
}

IdleAction MemoryBlock::release() {
    refCount--;
    if (refCount == 0) {
        return onLastRelease();
    }
    else {
        // מצב שבו refCount גדול מ 0
        return IdleAction::KeepCached;
    }
}

// --------- Getters : START ------------
int MemoryBlock::getRefCount() const {
    return refCount;
}
// --------- Getters : END --------------