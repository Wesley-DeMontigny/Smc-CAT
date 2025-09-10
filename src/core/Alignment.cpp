#include "Alignment.hpp"
#include <fstream>
#include <iostream>
#include <map>

Alignment::Alignment(std::string path) : numChar(0), numTaxa(0) {
    std::map<char, int> alphabet{
        {'A', 0},
        {'R', 1},
        {'N', 2},
        {'D', 3},
        {'C', 4},
        {'Q', 5},
        {'E', 6},
        {'G', 7},
        {'H', 8},
        {'I', 9},
        {'L', 10},
        {'K', 11},
        {'M', 12},
        {'F', 13},
        {'P', 14},
        {'S', 15},
        {'T', 16},
        {'W', 17},
        {'Y', 18},
        {'V', 19},
        {'-', 20},
    };

    
    std::ifstream file(path);

    std::vector<std::vector<int>> sequences{};

    int currentIndex = 0;
    if (file.is_open()){
        std::string line;
        std::vector<int> currentSequence;
        while(std::getline(file, line)){
            if(line[0] != '#'){ // Ignore comments
                if(line[0] == '>'){
                    if(currentIndex != 0){
                        sequences.push_back(currentSequence);
                        currentSequence.clear();
                    }

                    std::string name = "";
                    for(int i = 1; i < line.size(); i++){
                        name += line[i];
                    }
                    taxaNames.push_back(name);
                    currentIndex++;
                }
                else{
                    for(int i = 0; i < line.size(); i++){
                        char upper = std::toupper(line[i]);
                        if(alphabet.count(upper) > 0){
                            auto iter = alphabet.find(upper);
                            int code = iter->second;
                            currentSequence.push_back(code);
                        }
                        else {
                            if (std::isprint(upper) && !std::isspace(upper)) { 
                                std::cerr << "Warning: Detected unknown character in fasta (we are ignoring it): " << line[i] << std::endl;
                            }
                        }
                    }
                }
            }
        }
        file.close();
    }else{
        std::cerr << "Error: Unable to open alignment file!" << std::endl;
        std::exit(1);
    }

    numChar = sequences[0].size();
    numTaxa = taxaNames.size();
    for(int i = 1; i < sequences.size(); i++){
        int current_length = sequences[i].size();
        if(current_length != numChar){
            std::cerr << "Error: Sequence 1 had length " << numChar << " but sequence " << i+1 << " had length" << current_length << std::endl;
            std::exit(1);
        }
    }

    dataMatrix = std::unique_ptr<int[]>(new int[numChar * numTaxa]);
    for(int i = 0; i < numTaxa; i++){
        for(int j = 0; j < numChar; j++){
            dataMatrix[i*numChar + j] = sequences[i][j];
        }
    }
}