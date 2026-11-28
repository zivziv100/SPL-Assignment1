#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include "json.hpp"

#include <Server.h>
#include "Model.h"


using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(string configPath) : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), currentCycle(0), activeGpu(){
    Config conf = readConfig(configPath);
    gpus.reserve(conf.gpus.size());
    for (size_t i = 0; i < conf.gpus.size(); i++) {
        gpus.push_back(GPU(conf.gpus[i].memoryBytes / conf.blockSize, conf.blockSize));
    }
    
    models.reserve(conf.models.size());
    for (size_t i = 0; i < conf.models.size(); i++) {
        models.push_back(new Model(conf.models[i].id, conf.models[i].embeddingSize));
    }

    unordered_map<string, Model*> idToModelMap;
    for (size_t i = 0; i < models.size(); i++) {
        idToModelMap.insert(make_pair(models[i]->getModelId(), models[i]));
    }

    nonSubmittedRequests.reserve(conf.requests.size());
    for (size_t i = 0; i < conf.requests.size(); i++) {
        const RequestConfig& currentReq = conf.requests[i];
        const Model* currentModel = idToModelMap.at(currentReq.modelId);
        if (currentReq.type == RequestType::Generative) {
            nonSubmittedRequests.push_back(new GenerativeRequest(static_cast<int>(currentReq.id), *currentModel, currentReq.admissionCycle, currentReq.prompt, currentReq.scriptedOutput, currentReq.maxOutputTokens));

        }
        else {
            nonSubmittedRequests.push_back(new EmbeddingRequest(static_cast<int>(currentReq.id), *currentModel, currentReq.admissionCycle, currentReq.prompt));
        }
    }

    if (conf.policy == "arrivalOrder") policy = new ArrivalOrderPolicy();
    else if (conf.policy == "shortestJob") policy = new ShortestJobPolicy();
    else if (conf.policy == "longestPrefix") policy = new LongestPrefixPolicy();
}

// ------- RULE OF 5 : START -------
Server::~Server() {
    for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        delete nonSubmittedRequests[i];
    }
    nonSubmittedRequests.clear();
    for (size_t i = 0; i < waitingRequests.size(); i++) {
        delete waitingRequests[i];
    }
    waitingRequests.clear();
    for (size_t i = 0; i < activeRequests.size(); i++) {
        delete activeRequests[i];
    }
    activeRequests.clear();
    for (size_t i = 0; i < models.size(); i++) {
        delete models[i];
    }
    models.clear();
    if (policy != nullptr) {
        delete policy;
        policy = nullptr;
    }
}

Server::Server(const Server& other)
    : gpus(other.gpus),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      currentCycle(other.currentCycle),
      activeGpu(other.activeGpu) {
        
    if (other.policy != nullptr) {
        policy = other.policy->clone();
    }

    unordered_map<string, Model*> idToNewModel;
    for (size_t i = 0; i < other.models.size(); i++) {
        Model* copiedModel = new Model(*other.models[i]);
        models.push_back(copiedModel);
        idToNewModel.insert(make_pair(copiedModel->getModelId(), copiedModel));
    }

    for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
        string modelId = other.nonSubmittedRequests[i]->getModel().getModelId();
        const Model* matchingModel = idToNewModel.at(modelId);
        nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]->clone(*matchingModel));
    }

    for (size_t i = 0; i < other.waitingRequests.size(); i++) {
        string modelId = other.waitingRequests[i]->getModel().getModelId();
        const Model* matchingModel = idToNewModel.at(modelId);
        waitingRequests.push_back(other.waitingRequests[i]->clone(*matchingModel));
    }

    for (size_t i = 0; i < other.activeRequests.size(); i++) {
        string modelId = other.activeRequests[i]->getModel().getModelId();
        const Model* matchingModel = idToNewModel.at(modelId);
        activeRequests.push_back(other.activeRequests[i]->clone(*matchingModel));
    }
}
Server& Server::operator=(const Server& other) {
    if (this != &other) {
        for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
            delete nonSubmittedRequests[i];
        }
        nonSubmittedRequests.clear();

        for (size_t i = 0; i < waitingRequests.size(); i++) {
            delete waitingRequests[i];
        }
        waitingRequests.clear();

        for (size_t i = 0; i < activeRequests.size(); i++) {
            delete activeRequests[i];
        }
        activeRequests.clear();

        for (size_t i = 0; i < models.size(); i++) {
            delete models[i];
        }
        models.clear();

        if (policy != nullptr) {
            delete policy;
            policy = nullptr;
        }

        gpus = other.gpus;
        currentCycle = other.currentCycle;
        activeGpu = other.activeGpu;

        if (other.policy != nullptr) {
            policy = other.policy->clone();
        }

        unordered_map<string, Model*> idToNewModel;
        for (size_t i = 0; i < other.models.size(); i++) {
            Model* copiedModel = new Model(*other.models[i]);
            models.push_back(copiedModel);
            idToNewModel.insert(make_pair(copiedModel->getModelId(), copiedModel));
        }

        for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
            string modelId = other.nonSubmittedRequests[i]->getModel().getModelId();
            const Model* matchingModel = idToNewModel.at(modelId);
            nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]->clone(*matchingModel));
        }

        for (size_t i = 0; i < other.waitingRequests.size(); i++) {
            string modelId = other.waitingRequests[i]->getModel().getModelId();
            const Model* matchingModel = idToNewModel.at(modelId);
            waitingRequests.push_back(other.waitingRequests[i]->clone(*matchingModel));
        }

        for (size_t i = 0; i < other.activeRequests.size(); i++) {
            string modelId = other.activeRequests[i]->getModel().getModelId();
            const Model* matchingModel = idToNewModel.at(modelId);
            activeRequests.push_back(other.activeRequests[i]->clone(*matchingModel));
        }
    }

    return *this;
}

Server::Server(Server&& other) noexcept
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(other.policy),
      currentCycle(other.currentCycle),
      activeGpu(move(other.activeGpu)) {

    for (size_t i = 0; i < other.gpus.size(); i++) {
        gpus.push_back(move(other.gpus[i]));
    }
    other.gpus.clear();

    for (size_t i = 0; i < other.models.size(); i++) {
        models.push_back(other.models[i]);
    }
    other.models.clear();

    for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
        nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]);
    }
    other.nonSubmittedRequests.clear();

    for (size_t i = 0; i < other.waitingRequests.size(); i++) {
        waitingRequests.push_back(other.waitingRequests[i]);
    }
    other.waitingRequests.clear();

    for (size_t i = 0; i < other.activeRequests.size(); i++) {
        activeRequests.push_back(other.activeRequests[i]);
    }
    other.activeRequests.clear();
    other.activeGpu.clear();

    other.policy = nullptr;
    other.currentCycle = 0;
}
Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
            delete nonSubmittedRequests[i];
        }
        nonSubmittedRequests.clear();

        for (size_t i = 0; i < waitingRequests.size(); i++) {
            delete waitingRequests[i];
        }
        waitingRequests.clear();

        for (size_t i = 0; i < activeRequests.size(); i++) {
            delete activeRequests[i];
        }
        activeRequests.clear();

        for (size_t i = 0; i < models.size(); i++) {
            delete models[i];
        }
        models.clear();

        if (policy != nullptr) {
            delete policy;
            policy = nullptr;
        }
        gpus.clear();

        for (size_t i = 0; i < other.gpus.size(); i++) {
            gpus.push_back(move(other.gpus[i]));
        }
        other.gpus.clear();

        for (size_t i = 0; i < other.models.size(); i++) {
            models.push_back(other.models[i]);
        }
        other.models.clear();

        for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
            nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]);
        }
        other.nonSubmittedRequests.clear();

        for (size_t i = 0; i < other.waitingRequests.size(); i++) {
            waitingRequests.push_back(other.waitingRequests[i]);
        }
        other.waitingRequests.clear();

        for (size_t i = 0; i < other.activeRequests.size(); i++) {
            activeRequests.push_back(other.activeRequests[i]);
        }
        other.activeRequests.clear();

        policy = other.policy;
        currentCycle = other.currentCycle;
        activeGpu = move(other.activeGpu);

        other.policy = nullptr;
        other.currentCycle = 0;
        other.activeGpu.clear();
    }
    return *this;
}
// ------- RULE OF 5 : END ---------

void Server::serve() {
    while (step()) {}

    int totalHits = 0;
    int totalEvictions = 0;

    for (size_t i = 0; i < gpus.size(); i++) {
        totalHits += gpus[i].getCacheHits();
        totalEvictions += gpus[i].getEvictions();
    }

    cout << "summary: cycles=" << currentCycle
         << " cache_hits=" << totalHits
         << " evictions=" << totalEvictions << endl;
}
bool Server::step() {
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }

    arriveRequests();
    admitRequests();
    actRequests();
    currentCycle++;
    return true;
}

// -------- step() helper functions -------
void Server::arriveRequests() {
    vector<Request*> stillPending;
    for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        Request* req = nonSubmittedRequests[i];
        if (req->getAdmissionCycle() > currentCycle) { // Requests that their cycle is not now, go to pending
            stillPending.push_back(req);
            continue;
        }

        bool fitsSomewhere = false;
        for (size_t j = 0; j < gpus.size(); j++) {
            size_t blocks = gpus[j].getBlocksForMemory(req->getMaxNeededMemory()); // the request should fit in the gpu at her maximum needed memory
            if (blocks <= gpus[j].getBlocksNum()) { // if it fits, we stop searching
                fitsSomewhere = true;
                break;
            }
        }

        if (fitsSomewhere) { // if there is a valid GPU (the request fits in it), we send the request to waiting
            waitingRequests.push_back(req);
        }
        else { // otherwise, there is not GPU that the request fits in, so we reject and delete the request
            cout << "rejected " << req->getId() << " cycle=" << currentCycle << endl;
            delete req;
        }
    }
    nonSubmittedRequests = stillPending;
}
void Server::admitRequests() {
    if (waitingRequests.empty() || policy == nullptr) return;

    vector<const Request*> eligible(waitingRequests.begin(), waitingRequests.end());
    vector<int> order = policy->choose(eligible,gpus);

    vector<bool> admitted(waitingRequests.size(), false);
    for (size_t i = 0; i < order.size(); i++) {
        int index = order[i];
        if (tryAdmit(waitingRequests[index])) {
            admitted[index] = true;
        }
    }

    vector<Request*> stillWaiting;
    for (size_t i = 0; i < waitingRequests.size(); i++) {
        if (!admitted[i]) stillWaiting.push_back(waitingRequests[i]);
    }
    waitingRequests = stillWaiting;
}
bool Server::tryAdmit(Request* req) {
    int bestGpu = -1;
    size_t bestCached = 0;
    size_t bestNeed = 0;

    for (size_t g = 0; g < gpus.size(); g++) {
        PrefixMatch match = req->getCachedPrefix(gpus[g]);
        size_t cached = match.blockIds.size();

        size_t idleCached = 0;
        for (size_t i = 0; i < match.blockIds.size(); i++) {
            if (gpus[g].isIdle(match.blockIds[i])) idleCached++;
        }

        size_t need = gpus[g].getBlocksForMemory(req->getNeededMemory(gpus[g])) + idleCached;
        if (need > gpus[g].availableSlots()) continue;

        if (bestGpu == -1 || cached > bestCached || (cached == bestCached && need < bestNeed)) {
            bestGpu = static_cast<int>(g);
            bestCached = cached;
            bestNeed = need;
        }
    }
    if (bestGpu == -1) return false;


    size_t bestMatchLen = req->getCachedPrefix(gpus[bestGpu]).matchedTokens;
    int sourceGpu = -1;
    PrefixMatch sourceMatch;
    for (size_t g = 0; g < gpus.size(); g++) {
        if (static_cast<int>(g) == bestGpu) continue;
        
        PrefixMatch m = req->getCachedPrefix(gpus[g]);
        if (m.matchedTokens > bestMatchLen) {
            bestMatchLen = m.matchedTokens;
            sourceGpu = static_cast<int>(g);
            sourceMatch = m;
        }
    }
    if (sourceGpu != -1) { // if we found a GPU with bigger cache, we take copy from him
        gpus[bestGpu].copyMemoryFromOtherGPU(gpus[sourceGpu], sourceMatch.blockIds);
    }

    req->reserveMemory(gpus[bestGpu]);
    activeRequests.push_back(req);
    activeGpu.push_back(bestGpu);
    return true;
}
void Server::actRequests() {
    vector<Request*> stillActive;
    vector<int> stillActiveGpu;

    for (size_t i = 0; i < activeRequests.size(); i++) {
        Request* req = activeRequests[i];
        int g = activeGpu[i];

        if (!req->finished()) req->act(gpus[g]);

        if (req->finished()) {
            req->deallocateMemory(gpus[g]);
            cout << "finished " << req->getId() << " cycle=" << currentCycle << " gpu=" << g << ": " << req->getOutput() << endl;
            delete req;
        }
        else {
            stillActive.push_back(req);
            stillActiveGpu.push_back(g);
        }
    }
    activeRequests = stillActive;
    activeGpu = stillActiveGpu;
}
