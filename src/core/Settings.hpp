#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <iostream>

/*
    This struct loads in the user's settings from the command line and provides
    usage help.
*/
struct Settings {
    Settings(void);
    Settings(int argc, char* argv[]);

    int numParticles = 500;
    int rejuvenationIterations = 10;
    int numThreads = 8;
    bool invar = false;
    bool lg = false;
    double alg8Probability = 0.05;
    int numRates = 1;
    unsigned int seed = 1;
    std::string fastaFile = "/workspaces/FastCAT/local_testing/globin_aa_aligned.fasta";
};

#endif