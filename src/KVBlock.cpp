#include "MemoryBlock.h"

KVBlock::KVBlock(size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
   size_t newSize = size;
    KVBlock* newBlock = new KVBlock(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newBlock->writeToBlock(i, this->readFromBlock(i));
    }
    return newBlock;
}    