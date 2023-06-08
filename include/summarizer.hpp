#ifndef MODEL_H
#define MODEL_H

// #include <fstream>
// #include <string>
#include <vector>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <queue>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"


class Summarizer{
public:
    typedef std::vector<int> Sequence;
    typedef std::tuple<Sequence, double, void*> BeamElement;
    typedef std::vector<BeamElement> Beam;

    Summarizer();

    Sequence generate(Sequence input_ids_seq, Sequence attention_msk_seq, int num_beams=2, int max_length=60);


private:
    std::vector<std::vector<int>> _beamSearch(const std::vector<std::vector<float>>& logits, int beamWidth, int maxSteps, int maxLength, int startToken, int endToken);
    std::vector<std::vector<float>> _tensorToFloat2DVector(const Ort::Value& tensor, int rows, int columns);

    Ort::Env     _env;
    Ort::Session* _encoder_session;
    Ort::Session* _decoder_session;
};


#endif // MODEL_H
