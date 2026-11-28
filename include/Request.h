#ifndef REQUEST_H_
#define REQUEST_H_

#include <string>
#include <vector>
#include <GPU.h>

class Model;
class GPU;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    std::vector<std::string> tokens;
    std::vector<std::size_t> memoryBlocks;
    std::size_t reservedBlocks;

    // Rule of 3
    Request(const Request& other, const Model& model);
public:
    // Rule of 3
    Request(int id, const Model& model, int admissionCycle, std::string prompt);
    virtual ~Request();

    // Request Getters - for each type of request
    virtual std::size_t getNeededMemory(const GPU& gpu) const=0;
    virtual std::size_t getMaxNeededMemory() const=0;
    virtual std::vector<std::string> getCacheableTokens() const=0;
    virtual std::string getOutput() const = 0;
    virtual std::size_t getNeedToUnreserve() const = 0;

    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const = 0;
    virtual Request* clone(const Model& model) const = 0;
    virtual void act(GPU& gpu)=0;
    virtual bool finished() const=0;
    virtual void reserveMemory(GPU& gpu) = 0;
    
    void deallocateMemory(GPU& gpu);
    
    // Getters
    int getId() const;
    const Model& getModel() const;
    int getAdmissionCycle() const;
    PrefixMatch getCachedPrefix(const GPU& gpu) const;
};

class GenerativeRequest: public Request{
private:
    std::string scriptedOutput;
    std::size_t maxOutputTokens;
    std::vector<std::string> tokensOutput;
    std::size_t tokensInMemory;

    GenerativeRequest(const GenerativeRequest& other, const Model& model);
    std::vector<std::string> getSequencePrefix(std::size_t length) const;
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens);

    virtual ~GenerativeRequest();

    virtual std::size_t getNeededMemory(const GPU& gpu) const override;
    virtual std::size_t getMaxNeededMemory() const override;
    virtual std::vector<std::string> getCacheableTokens() const override;
    virtual std::string getOutput() const override;
    virtual std::size_t getNeedToUnreserve() const override;

    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    virtual Request* clone(const Model& model) const override;
    virtual void act(GPU& gpu) override;
    virtual bool finished() const override;
    virtual void reserveMemory(GPU& gpu) override;
};

class EmbeddingRequest: public Request{

private:
    std::size_t currentTokenIndex;
    std::vector<int> result;

    // Rule of 3
    EmbeddingRequest(const EmbeddingRequest& other, const Model& model);
public:
    // Rule of 3
    EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt);
    virtual ~EmbeddingRequest();

    virtual std::size_t getNeededMemory(const GPU& gpu) const override;
    virtual std::size_t getMaxNeededMemory() const override;
    virtual std::vector<std::string> getCacheableTokens() const override;
    virtual std::string getOutput() const override;
    virtual std::size_t getNeedToUnreserve() const override;
    
    
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const override;
    virtual Request* clone(const Model& model) const override;
    virtual void act(GPU& gpu) override;
    virtual bool finished() const override;
    virtual void reserveMemory(GPU& gpu) override;

    std::size_t getSlot(size_t k, const GPU& gpu) const;
    std::size_t getOffset(size_t k, const GPU& gpu) const;
};
#endif