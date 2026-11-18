#include <fstream>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"
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

Server::Server(string configPath) : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr), cacheHits(0), evictions(0), currentCycle(0){
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