#include <iostream>
#include <string>
#include <vector>

#include "tokenizer.hpp"
#include "summarizer.hpp"


void displayUsage() {
    std::cout << "Usage: program [-p] [-h]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << " No flag given runs in default mode " << std::endl;
    std::cout << "  -p      Profile Mode" << std::endl;
    std::cout << "  -h      Display usage information" << std::endl;
}


int main(int argc, char* argv[]) {
    bool profile_mode = false;

    // Checking if the flag is provided
    if (argc > 1) {
        std::string flag = argv[1];

        // Checking the flag
        if (flag == "-p") {
            std::cout << "Running Profiling Mode" << std::endl;
            profile_mode = true;
        } else {
            displayUsage();
            return 0;
        }
    }
    // Load the Tokenizer Class
    auto tokenizer = TokenizergRPC();
    Summarizer model;


    if(profile_mode) {
        // Run through using a default input for profiling
        std::string input_sentence = "CNN)Governments around the world are using the threat of terrorism -- real or perceived -- to advance executions, Amnesty International alleges in its annual report on the death penalty. \"The dark trend of governments using the death penalty in a futile attempt to tackle real or imaginary threats to state security and public safety was stark last year,\" said Salil Shetty, Amnesty's Secretary General in a release. \"It is shameful that so many states around the world are essentially playing with people's lives -- putting people to death for 'terrorism' or to quell internal instability on the ill-conceived premise of deterrence.\" The report, \"Death Sentences and Executions 2014,\" cites the example of Pakistan lifting a six-year moratorium on the execution of civilians following the horrific attack on a school in Peshawar in December. China is also mentioned, as having used the death penalty as a tool in its \"Strike Hard\" campaign against terrorism in the restive far-western province of Xinjiang";
        auto tokens = tokenizer.tokenize(input_sentence);
        auto seq = model.generate(tokens.input_ids, tokens.attention_mask, 2, 60);
        std::string decoded = tokenizer.decode(seq);
        std::cout << decoded << std::endl;
        return 0;
    } 


    

    return 0;
}