#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

#include "tokenize.grpc.pb.h"

class TokenizergRPC {
public:
    // Constructor
    explicit TokenizergRPC();

    struct tokenizedOutput{
        std::vector<int> input_ids;
        std::vector<int> attention_mask;
    };

    // Tokenize a string
    tokenizedOutput tokenize(const std::string& inputString) const;

    // Decode a string
    std::string decode(const std::vector<int>& tokens) const;

private:
    std::unique_ptr<TokenizerRPC::Stub> _stub;

};

#endif // TOKENIZER_H