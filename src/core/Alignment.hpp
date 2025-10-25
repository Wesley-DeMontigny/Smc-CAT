#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP
#include <string>
#include <memory>
#include <vector>

/**
 * @brief Loads in data from a FASTA File and converts it into a matrix encoding the index of each character.
 */
class Alignment {
    public:
        Alignment(void) = delete;
        Alignment(std::string path);
        int operator()(int t, int s) {return dataMatrix[t*numChar + s];}
        int getNumChar() {return numChar;}
        int getNumTaxa() {return numTaxa;}
        std::vector<std::string> getTaxaNames() {return taxaNames;}
    private:
        int numChar;
        int numTaxa;
        std::vector<std::string> taxaNames;
        std::unique_ptr<int[]> dataMatrix;
};

#endif