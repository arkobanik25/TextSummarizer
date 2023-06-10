#include "tokenizer.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <grpcpp/grpcpp.h>


TokenizergRPC::TokenizergRPC() {
    // Create gRPC channel
    auto channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
    _stub = TokenizerRPC::NewStub(channel);
}

//Tokenize a sentence using the loaded vocabulary
TokenizergRPC::tokenizedOutput TokenizergRPC::tokenize(const std::string& inputString) const {
    TokenizergRPC::tokenizedOutput tokens;

    TokenizeRequest request;
    TokenizeResponse reply;
    grpc::ClientContext context;

    request.set_text(inputString);

    // Call the request
    grpc::Status status = _stub->Tokenize(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
        const google::protobuf::RepeatedField<int32_t>& tokensData = reply.tokens();
        std::vector<int32_t> tokensVector(tokensData.begin(), tokensData.end());
        tokens.input_ids = tokensVector;

        const google::protobuf::RepeatedField<int32_t>& attentionMaskData = reply.attention_mask();
        std::vector<int32_t> attentionMaskVector(attentionMaskData.begin(), attentionMaskData.end());

        tokens.attention_mask = attentionMaskVector;
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::cout << "RPC failed" << std::endl;
    }


    return tokens;
}

// Decode a sequence of token IDs into text using the vocabulary
std::string TokenizergRPC::decode(const std::vector<int>& tokens) const{
    std::string text;

    DecodeRequest request;
    DecodeResponse reply;

    grpc::ClientContext context;

    request.mutable_tokens()->CopyFrom({tokens.begin(), tokens.end()});

    // Call the request
    grpc::Status status = _stub->Decode(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
        text = reply.text();
    } else {
        std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
        std::cout << "RPC failed" << std::endl;
    }

    return text;
}