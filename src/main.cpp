#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <chrono>
#include <malloc.h>

#include "tokenizer.hpp"
#include "summarizer.hpp"

/**************************************************************** Helper Functions ********************************************************************/

void displayUsage() {
    std::cout << "Usage: program [-p] [-h]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << " No flag given runs in default mode " << std::endl;
    std::cout << "  -p      Profile Mode" << std::endl;
    std::cout << "  -h      Display usage information" << std::endl;
}

void displayMainOptions() {
    std::cout <<"Please Choose One of the Following Options:" << std::endl;
    std::cout <<"    1. Enter Text." << std::endl;
    std::cout <<"    2. Choose .txt file." << std::endl;
    std::cout <<"    3. Exit." << std::endl;
    std::cout << ">> ";
}

bool isValidChoice(const std::string& input) {
    return (input == "1" || input == "2" || input == "3");
}

bool isTextFile(const std::string& filename) {
    std::filesystem::path filePath(filename);
    return (std::filesystem::exists(filePath) && filePath.extension() == ".txt");
}

std::string readFileToString(const std::string& filename) {
    std::ifstream file(filename);
    std::string fileContents;

    if (file) {
        fileContents.assign((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
    }

    return fileContents;
}

/********************************************************************** Main *************************************************************************/

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

    if(profile_mode) {
        // Run through using a default input for profiling
        std::string input_sentence = "CNN)Governments around the world are using the threat of terrorism -- real or perceived -- to advance executions, Amnesty International alleges in its annual report on the death penalty. \"The dark trend of governments using the death penalty in a futile attempt to tackle real or imaginary threats to state security and public safety was stark last year,\" said Salil Shetty, Amnesty's Secretary General in a release. \"It is shameful that so many states around the world are essentially playing with people's lives -- putting people to death for 'terrorism' or to quell internal instability on the ill-conceived premise of deterrence.\" The report, \"Death Sentences and Executions 2014,\" cites the example of Pakistan lifting a six-year moratorium on the execution of civilians following the horrific attack on a school in Peshawar in December. China is also mentioned, as having used the death penalty as a tool in its \"Strike Hard\" campaign against terrorism in the restive far-western province of Xinjiang";
        auto tokens = tokenizer.tokenize(input_sentence);

        struct mallinfo before = mallinfo();
        Summarizer model;

        auto startTime = std::chrono::high_resolution_clock::now();
        auto seq = model.generate(tokens.input_ids, tokens.attention_mask, 2, 60);
        auto endTime = std::chrono::high_resolution_clock::now();
        struct mallinfo after = mallinfo();
        int memoryUsed = after.uordblks - before.uordblks;
        std::cout << "Memory used by the ONNX Model: " << memoryUsed << " bytes" << std::endl;

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Generation Execution time: " << duration.count() << " milliseconds" << std::endl;

        std::string decoded = tokenizer.decode(seq);
        std::cout << "Generated Sumamry: \n"  << decoded << std::endl;
        return 0;
    } 

    // Load the Model
    Summarizer model;

    std::string input;
    bool exit = false;
    while(1) {
        bool valid = false;
        displayMainOptions();
        std::getline(std::cin, input);
        if (isValidChoice(input)) {
            int choice = std::stoi(input);
            std::string input_text;
            switch(choice) {
                case 1:
                    std::cout << "Enter Text to be Summarized:" << std::endl;
                    std::cout <<">> ";
                    std::getline(std::cin, input_text);
                    valid = true;
                    break;

                case 2: 
                    std::cout << "Enter a file name: " << std::endl;
                    std::cout <<">> ";
                    std::getline(std::cin, input_text);
                    if(isTextFile(input_text)) {
                        input_text = readFileToString(input_text);
                        valid = true;
                    } else {
                        std::cout << "File either does not exist or is not a .txt File\n\n" << std::endl;
                    }
                    break;
                
                case 3:
                    exit = true;
                    break;

                default:
                    break;
            }
            if(valid) {
                auto tokens = tokenizer.tokenize(input_text);
                auto seq = model.generate(tokens.input_ids, tokens.attention_mask, 2, 60);
                std::string generated_string = tokenizer.decode(seq);
                std::cout << "Generated Sumamry: \n" << generated_string << "\n\n"<< std::endl;
            } 
            
        } else {
            std::cout << "Invalid input." << std::endl;
        }
        if(exit) break;
    }
    std::cout << "Exiting the program." << std::endl;

    return 0;
}