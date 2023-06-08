#include "summarizer.hpp"

#include <queue>
#include <algorithm>
#include <cmath>

Summarizer::Summarizer():_env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel") {

    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    _encoder_session = new Ort::Session(_env, "encoder_model.onnx", session_options);
    _decoder_session = new Ort::Session(_env, "decoder_model.onnx", session_options);

}

std::vector<std::vector<float>> Summarizer::_tensorToFloat2DVector(const Ort::Value& tensor, int rows, int columns) {
    // Check tensor data type
    if (tensor.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Tensor data type is not float.");
    }

    // Get tensor data pointer and total element count
    const float* tensorData = tensor.GetTensorData<float>();
    size_t totalElementCount = tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    // Check if the total element count matches the desired size
    if (totalElementCount != rows * columns) {
        throw std::runtime_error("Tensor size does not match the desired size.");
    }

    // Convert tensor to 2D float vector
    std::vector<std::vector<float>> float2DVector(rows, std::vector<float>(columns));
    size_t elementIndex = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            float2DVector[i][j] = tensorData[elementIndex];
            elementIndex++;
        }
    }

    return float2DVector;
}


std::vector<std::vector<int>> Summarizer::_beamSearch(const std::vector<std::vector<float>>& logits, int beamWidth, int maxSteps, int maxLength, int startToken, int endToken) {
    // Number of tokens in the vocabulary
    int vocabSize = logits[0].size();

    // Initialize the beam search
    std::vector<std::vector<int>> beams(beamWidth, std::vector<int>(1, startToken));
    std::vector<float> beamScores(beamWidth, 0.0);

    // Iterate over the maximum number of steps or until maxLength is reached
    for (int step = 0; step < maxSteps && step < maxLength; step++) {
        // Create a priority queue to hold the top beam candidates
        std::priority_queue<std::pair<float, std::vector<int>>> candidates;

        // Generate new candidates for each beam
        for (int beamIndex = 0; beamIndex < beamWidth; beamIndex++) {
            const std::vector<int>& beam = beams[beamIndex];
            float beamScore = beamScores[beamIndex];

            // Check if the beam ends with the end token
            if (beam.back() == endToken) {
                candidates.push(std::make_pair(beamScore, beam));
                continue;
            }

            // Get the logits for the next token
            const std::vector<float>& stepLogits = logits[step];

            // Generate candidates for the next token
            for (int token = 0; token < vocabSize; token++) {
                std::vector<int> candidate = beam;
                candidate.push_back(token);
                float candidateScore = beamScore + stepLogits[token];
                candidates.push(std::make_pair(candidateScore, candidate));
            }
        }

        // Select the top candidates to form the new beams
        std::vector<std::vector<int>> newBeams;
        std::vector<float> newBeamScores;
        for (int i = 0; i < beamWidth; i++) {
            if (candidates.empty()) {
                break;
            }
            const auto& candidate = candidates.top();
            newBeams.push_back(candidate.second);
            newBeamScores.push_back(candidate.first);
            candidates.pop();
        }

        // Stop the search if all the new beams end with the end token
        bool allEndTokens = true;
        for (const auto& beam : newBeams) {
            if (beam.back() != endToken) {
                allEndTokens = false;
                break;
            }
        }
        if (allEndTokens) {
            return newBeams;
        }

        // Update the beams and scores for the next step
        beams = newBeams;
        beamScores = newBeamScores;
    }

    return beams;
}


Summarizer::Sequence Summarizer::generate(Sequence input_ids_seq, 
                                          Sequence attention_msk_seq, 
                                          int num_beams, 
                                          int max_length) {

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 128};
    std::vector<int64_t> attention_shape = {1, 128}; 

    std::vector<int64_t> input_ids(input_ids_seq.begin(), input_ids_seq.end());
    std::vector<int64_t> attention_mask(attention_msk_seq.begin(), attention_msk_seq.end());
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), attention_shape.data(), attention_shape.size());

    std::vector<const char*> enc_input_names = {"input_ids", "attention_mask"};
    std::vector<const char*> enc_output_names = {"last_hidden_state"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_ids_tensor));
    input_tensors.push_back(std::move(attention_mask_tensor));

    std::vector<Ort::Value> last_hidden_state_tensors = _encoder_session->Run(Ort::RunOptions{nullptr}, enc_input_names.data(), input_tensors.data(), enc_input_names.size(), enc_output_names.data(), enc_output_names.size());

    std::vector<const char*> dec_input_names = {"input_ids", "encoder_hidden_states"};
    std::vector<const char*> dec_output_names = {"logits"};
    std::vector<Ort::Value> middle_tensors;
    input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    middle_tensors.push_back(std::move(input_ids_tensor));
    middle_tensors.push_back(std::move(last_hidden_state_tensors[0]));

    std::vector<Ort::Value> logits_tensor = _decoder_session->Run(Ort::RunOptions{nullptr}, dec_input_names.data(), middle_tensors.data(), dec_input_names.size(), dec_output_names.data(), dec_output_names.size());

    std::vector<std::vector<float>> logit_vector = _tensorToFloat2DVector(logits_tensor[0], 128, 50264);

    auto seq = _beamSearch(logit_vector, num_beams, 60, max_length, 0, 2);

    return seq[0];
}