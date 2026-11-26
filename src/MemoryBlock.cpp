#include "MemoryBlock.h"
#include <algorithm>   
#include <stdexcept>   

MemoryBlock::MemoryBlock(size_t size) :
 size(size), refCount(0), data(new int[size])  
 {
    for (size_t i = 0; i < size; i++) {
    data[i] = 0;
 }
}

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

int MemoryBlock::getRefCount() const {
    return refCount;
}
