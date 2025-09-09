#ifndef ALIGNMENT_HPP
#define ALIGNMENT_HPP
#include <iostream>

/*
    Loads in data from a FASTA File and converts it into an ambiguity-aware data format based on
    bit flags.
*/
class Alignment {
    public:
        Alignment(void) = delete;
        Alignment(std::string path);
    private:
};

#endif