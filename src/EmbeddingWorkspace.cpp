#include "MemoryBlock.h"

EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    size_t newSize = size;
    EmbeddingWorkspace* newBlock = new EmbeddingWorkspace(newSize);
    for (size_t i = 0; i < newSize; ++i) {
        newBlock->writeToBlock(i, this->readFromBlock(i));
    }
    return newBlock;
}